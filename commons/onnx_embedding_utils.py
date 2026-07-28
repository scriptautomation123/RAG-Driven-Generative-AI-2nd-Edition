"""
onnx_embedding_utils.py

Reusable helpers for calling the Oracle 26ai in-database ONNX embedding model
(sentence-transformers/all-MiniLM-L6-v2, 384-dim FLOAT32).

Import this module in notebooks instead of calling OpenAI for embeddings.
LLM generation should continue to use the OpenAI client as before.
"""

import oracledb


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_VECTOR_TYPE = "VECTOR(384, FLOAT32)"


def get_embedding(cursor, text):
    """
    Generate a single embedding vector via the PL/SQL GET_EMBEDDING function.

    Parameters
    ----------
    cursor : oracledb.Cursor
        An open Oracle cursor. The caller owns the connection lifecycle.
    text : str
        Text to embed.

    Returns
    -------
    list[float]
        384-dimensional FLOAT32 embedding vector.
    """
    cursor.setinputsizes(text=oracledb.DB_TYPE_VECTOR)
    cursor.execute("SELECT GET_EMBEDDING(:text) FROM DUAL", {"text": text})
    row = cursor.fetchone()
    if row is None or row[0] is None:
        raise RuntimeError("GET_EMBEDDING returned no vector")
    return list(row[0])


def get_embeddings_batch(cursor, texts):
    """
    Generate embeddings for multiple texts.

    Oracle 26ai currently exposes GET_EMBEDDING as a scalar PL/SQL function,
    so this implementation iterates over the input list and falls back to
    individual calls. If a future release supports batched ONNX scoring inside
    SQL, replace the loop with a single bulk execute/fetch.

    Parameters
    ----------
    cursor : oracledb.Cursor
        An open Oracle cursor.
    texts : list[str]
        Texts to embed.

    Returns
    -------
    list[list[float]]
        List of 384-dimensional FLOAT32 embedding vectors, in the same order
        as `texts`.
    """
    if not texts:
        return []

    # Bind the vector input type once for the scalar calls.
    cursor.setinputsizes(text=oracledb.DB_TYPE_VECTOR)

    vectors = []
    # Using prepare + execute many times lets the database reuse the cursor;
    # the input size hint remains effective across executions.
    cursor.prepare("SELECT GET_EMBEDDING(:text) FROM DUAL")
    for text in texts:
        cursor.execute(None, {"text": text})
        row = cursor.fetchone()
        if row is None or row[0] is None:
            raise RuntimeError("GET_EMBEDDING returned no vector")
        vectors.append(list(row[0]))
    return vectors
