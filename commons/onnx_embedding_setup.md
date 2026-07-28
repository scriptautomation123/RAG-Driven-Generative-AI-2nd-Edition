# In-Database ONNX Embeddings with Oracle 26ai

This repository now uses **data-sovereign embeddings** powered by
`sentence-transformers/all-MiniLM-L6-v2` running directly inside an
**Oracle 26ai Free Tier** database. The model is stored as an ONNX artifact
and invoked from pure SQL/PLSQL, so no OpenAI API calls are required for the
embedding layer. LLM generation still uses OpenAI (`gpt-*`) as before.

## Why in-database embeddings?

- **Data sovereignty:** raw text never leaves the database to reach an external embedding provider.
- **Cost reduction:** no per-token embedding charges.
- **Latency:** embedding and vector search happen in the same SQL execution context.
- **Dimensionality:** `all-MiniLM-L6-v2` produces 384-dimensional `FLOAT32` vectors, so all tables declare `VECTOR(384, FLOAT32)`.

## Model overview

| Property | Value |
|----------|-------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Output dimension | 384 |
| Output type | `FLOAT32` |
| Oracle vector type | `VECTOR(384, FLOAT32)` |
| Mean pooling / normalization | handled inside the exported ONNX graph |

## 1. Export the model to ONNX

Use the `optimum` CLI (recommended) or export manually with
`sentence-transformers` + `onnx`.

### Option A: `optimum[onnxruntime]` (recommended)

```bash
pip install optimum[onnxruntime] sentence-transformers
optimum-cli export onnx \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --task feature-extraction \
  --optimize O2 \
  ./all-MiniLM-L6-v2-onnx
```

The exported directory will contain `model.onnx`, `config.json`, and tokenizer files.

### Option B: Manual export with `sentence-transformers` + `onnx`

```python
from pathlib import Path
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model_path = Path("all-MiniLM-L6-v2-onnx")
model_path.mkdir(exist_ok=True)

# Exports model.onnx, config.json and tokenizer files.
model.save(str(model_path), "onnx")
```

### Verify the ONNX output shape

```python
import onnx
m = onnx.load("all-MiniLM-L6-v2-onnx/model.onnx")
print(onnx.helper.printable_graph(m.graph))   # output should be [-1, 384]
```

## 2. Import the ONNX model into Oracle 26ai

Log in as a user with `CREATE MINING MODEL` privilege (e.g. `ADMIN` or an
application user) and run:

```sql
BEGIN
  -- Load the ONNX BLOB from a directory or from Python via oracledb.
  -- Below is the PL/SQL pattern assuming p_onnx_blob is a BLOB variable.
  DBMS_DATA_MINING.IMPORT_ONNX_MODEL(
    model_name  => 'MINILM_L6_V2_EMBEDDING',
    model_data  => :p_onnx_blob,
    metadata    => JSON_OBJECT('function' VALUE 'embedding',
                               'input'    VALUE 'text',
                               'output'   VALUE 'vector')
  );
END;
/
```

### Loading the BLOB from Python

```python
import oracledb

with open("all-MiniLM-L6-v2-onnx/model.onnx", "rb") as f:
    onnx_blob = f.read()

with oracledb.connect(user="USER", ******, dsn="HOST:PORT/DBNAME") as conn:
    with conn.cursor() as cursor:
        cursor.setinputsizes(p_onnx_blob=oracledb.DB_TYPE_BLOB)
        cursor.execute("""
            BEGIN
              DBMS_DATA_MINING.IMPORT_ONNX_MODEL(
                model_name  => 'MINILM_L6_V2_EMBEDDING',
                model_data  => :p_onnx_blob,
                metadata    => JSON_OBJECT('function' VALUE 'embedding',
                                           'input'    VALUE 'text',
                                           'output'   VALUE 'vector')
              );
            END;
        """, {"p_onnx_blob": onnx_blob})
    conn.commit()
```

## 3. PL/SQL helper: GET_EMBEDDING

Create a wrapper function that tokenizes the input text, invokes the ONNX model,
and returns a `VECTOR(384, FLOAT32)`.

```sql
CREATE OR REPLACE FUNCTION GET_EMBEDDING(
  p_text IN VARCHAR2
) RETURN VECTOR
AS
  v_vector  VECTOR(384, FLOAT32);
  v_result  CLOB;
  v_json    JSON_OBJECT_T;
BEGIN
  -- Oracle 26ai can tokenize and score the ONNX model in a single call.
  -- The result is returned as a JSON array of floats.
  SELECT PREDICTION(MINILM_L6_V2_EMBEDDING USING p_text AS text)
    INTO v_result
    FROM DUAL;

  v_json := JSON_OBJECT_T(v_result);
  v_vector := VECTOR(v_json.get_Array('output'), 384, FLOAT32);
  RETURN v_vector;
END GET_EMBEDDING;
/
```

If your Oracle release returns the prediction as a plain vector instead of JSON,
the function can be simplified to:

```sql
CREATE OR REPLACE FUNCTION GET_EMBEDDING(
  p_text IN VARCHAR2
) RETURN VECTOR
AS
  v_vector VECTOR(384, FLOAT32);
BEGIN
  SELECT PREDICTION(MINILM_L6_V2_EMBEDDING USING p_text AS text)
    INTO v_vector
    FROM DUAL;
  RETURN v_vector;
END GET_EMBEDDING;
/
```

Grant execute to the application schema if needed:

```sql
GRANT EXECUTE ON GET_EMBEDDING TO APPLICATION_USER;
```

## 4. Use the embedding from Python with `oracledb`

```python
import oracledb

oracledb.defaults.fetch_lobs = False

with oracledb.connect(user="USER", ******, dsn="HOST:PORT/DBNAME") as conn:
    with conn.cursor() as cursor:
        cursor.setinputsizes(text=oracledb.DB_TYPE_VECTOR)
        cursor.execute(
            "SELECT GET_EMBEDDING(:text) FROM DUAL",
            {"text": "In-database embeddings are sovereign."}
        )
        vec, = cursor.fetchone()
        print(len(vec), vec[:5])  # 384 floats
```

For notebooks in this repository, use the helper in
`commons/onnx_embedding_utils.py`:

```python
import sys
sys.path.append("../commons")
from onnx_embedding_utils import get_embedding, get_embeddings_batch

# single text
emb = get_embedding(cursor, "Oracle 26ai in-database ONNX embedding")

# batch of texts
embs = get_embeddings_batch(cursor, ["chunk one", "chunk two", "chunk three"])
```

## 5. Updating table DDL

All vector columns that previously stored OpenAI `text-embedding-3-small`
vectors now use:

```sql
embedding VECTOR(384, FLOAT32)
```

The `commons/create_script.py` registry and the DBA notebooks have been updated
accordingly. Existing data must be re-embedded with `GET_EMBEDDING` after the
column type is changed.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ORA-38453` / vector dimension mismatch | Column declared `VECTOR(1536)` or wrong output size | Use `VECTOR(384, FLOAT32)` |
| `ORA-20001` importing ONNX | Model already exists | `DROP MINING MODEL MINILM_L6_V2_EMBEDDING;` and re-import |
| Empty / unexpected vector shape | Tokenization mismatch in ONNX graph | Re-export with `task=feature-extraction` and mean pooling enabled |
| Python returns `None` for vector | `oracledb` vector type not bound | Use `cursor.setinputsizes(...=oracledb.DB_TYPE_VECTOR)` |

## References

- Oracle 26ai documentation: `DBMS_DATA_MINING.IMPORT_ONNX_MODEL`
- Hugging Face: `sentence-transformers/all-MiniLM-L6-v2`
- Optimum ONNX export: https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/pytorch
