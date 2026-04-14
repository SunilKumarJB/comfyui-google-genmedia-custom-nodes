# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sqlite3
import numpy as np
from google.cloud import storage
from .logger import get_node_logger

logger = get_node_logger(__name__)

DB_DIR = os.path.join("output", "asset_manager")
DB_PATH = os.path.join(DB_DIR, "assets.db")


def get_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT UNIQUE NOT NULL,
            filetype TEXT NOT NULL,
            tags TEXT,
            caption TEXT,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()


def insert_or_update_asset(filepath, filetype, tags, caption, embedding_list):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    tags_str = ", ".join(tags) if isinstance(tags, (list, tuple)) else str(tags or "")

    embedding_blob = None
    if embedding_list and len(embedding_list) > 0:
        arr = np.array(embedding_list, dtype=np.float32)
        embedding_blob = arr.tobytes()

    cursor.execute(
        """
        INSERT INTO assets (filepath, filetype, tags, caption, embedding)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(filepath) DO UPDATE SET
            tags=excluded.tags,
            caption=excluded.caption,
            embedding=excluded.embedding,
            created_at=CURRENT_TIMESTAMP
    """,
        (filepath, filetype, tags_str, caption, embedding_blob),
    )
    conn.commit()
    conn.close()


def insert_asset_cloud(file_bytes, filename, filetype, tags, caption, embedding_list, gcs_bucket, bq_dataset, bq_table, project_id=None):
    """
    Uploads media to GCS and stores metadata/embeddings into BigQuery.
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        raise RuntimeError("google-cloud-bigquery package is not installed. Required for Cloud mode.")

    # 1. GCS Upload
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(gcs_bucket)
    if not bucket.exists():
        logger.info(f"Creating GCS bucket: {gcs_bucket}")
        bucket = storage_client.create_bucket(gcs_bucket)

    gcs_path = f"assets/{filename}"
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(file_bytes, content_type=filetype)
    gcs_uri = f"gs://{gcs_bucket}/{gcs_path}"
    logger.info(f"Uploaded media to {gcs_uri}")

    # 2. BQ Write
    bq_client = bigquery.Client(project=project_id)
    dataset_ref = bq_client.dataset(bq_dataset)

    # Create dataset if not exists
    try:
        bq_client.get_dataset(dataset_ref)
    except Exception:
        logger.info(f"Creating BigQuery dataset: {bq_dataset}")
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        bq_client.create_dataset(dataset, exists_ok=True)

    table_ref = dataset_ref.table(bq_table)
    
    # Define schema for BQ table
    schema = [
        bigquery.SchemaField("filepath", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("filetype", "STRING"),
        bigquery.SchemaField("tags", "STRING"),
        bigquery.SchemaField("caption", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"),
    ]

    # Create table if not exists
    try:
        bq_client.get_table(table_ref)
    except Exception:
        logger.info(f"Creating BigQuery table: {bq_table}")
        table = bigquery.Table(table_ref, schema=schema)
        bq_client.create_table(table, exists_ok=True)

    # Insert row
    tags_str = ", ".join(tags) if isinstance(tags, (list, tuple)) else str(tags or "")
    emb_data = [float(v) for v in embedding_list] if embedding_list is not None else []

    row = {
        "filepath": gcs_uri,
        "filetype": filetype,
        "tags": tags_str,
        "caption": caption,
        "embedding": emb_data,
    }

    errors = bq_client.insert_rows_json(table_ref, [row])
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")

    logger.info(f"Stored metadata in BigQuery {bq_dataset}.{bq_table}")


def get_all_assets(limit=100, offset=0, storage_mode="local", bq_dataset="comfyui_assets", bq_table="media_index", project_id=None):
    if storage_mode in ["gcs_bq", "cloud"]:
        try:
            from google.cloud import bigquery
            client = bigquery.Client(project=project_id)
            query = f"SELECT filepath, filetype, tags, caption FROM `{bq_dataset}.{bq_table}` LIMIT {limit} OFFSET {offset}"
            rows = client.query(query).result()
            return [{"id": i, "filepath": r["filepath"], "filetype": r["filetype"], "tags": r["tags"], "caption": r["caption"]} for i, r in enumerate(rows)]
        except Exception as e:
            logger.error(f"Error fetching from BigQuery: {e}")
            return []
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, filepath, filetype, tags, caption, created_at FROM assets ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def search_assets_by_tags(query_tags, storage_mode="local", bq_dataset="comfyui_assets", bq_table="media_index", project_id=None):
    if storage_mode in ["gcs_bq", "cloud"]:
        try:
            from google.cloud import bigquery
            client = bigquery.Client(project=project_id)
            conditions = [f"tags LIKE '%{t.strip()}%'" for t in query_tags]
            where_clause = " OR ".join(conditions) if conditions else "1=1"
            query = f"SELECT filepath, filetype, tags, caption FROM `{bq_dataset}.{bq_table}` WHERE {where_clause} LIMIT 50"
            rows = client.query(query).result()
            return [{"id": i, "filepath": r["filepath"], "filetype": r["filetype"], "tags": r["tags"], "caption": r["caption"]} for i, r in enumerate(rows)]
        except Exception as e:
            logger.error(f"BigQuery Tag Search Error: {e}")
            return []
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    likes = []
    params = []
    for t in query_tags:
        likes.append("tags LIKE ?")
        params.append(f"%{t.strip()}%")

    where_clause = " OR ".join(likes) if likes else "1=1"
    cursor.execute(
        f"SELECT id, filepath, filetype, tags, caption, created_at FROM assets WHERE {where_clause} ORDER BY created_at DESC LIMIT 50",
        params,
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_similar_assets(query_embedding, top_k=10, storage_mode="local", bq_dataset="comfyui_assets", bq_table="media_index", project_id=None):
    query_arr = np.array(query_embedding, dtype=np.float32)
    results = []

    if storage_mode in ["gcs_bq", "cloud"]:
        try:
            from google.cloud import bigquery
            client = bigquery.Client(project=project_id)
            query = f"SELECT filepath, filetype, tags, caption, embedding FROM `{bq_dataset}.{bq_table}`"
            rows = client.query(query).result()
            for i, row in enumerate(rows):
                emb_list = row.get("embedding", [])
                if not emb_list:
                    continue
                db_arr = np.array(emb_list, dtype=np.float32)
                if db_arr.shape == query_arr.shape:
                    sim = float(cosine_similarity(query_arr, db_arr))
                    results.append({
                        "id": i,
                        "filepath": row["filepath"],
                        "filetype": row["filetype"],
                        "tags": row["tags"],
                        "caption": row["caption"],
                        "similarity": sim
                    })
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"BigQuery Semantic Search Error: {e}")
            return []
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, filepath, filetype, tags, caption, embedding FROM assets WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        blob = row["embedding"]
        if not blob:
            continue
        db_arr = np.frombuffer(blob, dtype=np.float32)
        if db_arr.shape == query_arr.shape:
            sim = float(cosine_similarity(query_arr, db_arr))
            results.append(
                {
                    "id": row["id"],
                    "filepath": row["filepath"],
                    "filetype": row["filetype"],
                    "tags": row["tags"],
                    "caption": row["caption"],
                    "similarity": sim,
                }
            )

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]
