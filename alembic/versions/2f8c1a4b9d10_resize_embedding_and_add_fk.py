"""resize embedding vector and add document foreign key

Revision ID: 2f8c1a4b9d10
Revises: 69aaa3efd2c7
Create Date: 2026-10-20

The first migration created `embedding` as vector(768), but the configured
embedding model (gemini-embedding-2) returns 3072-dimensional vectors. This
migration aligns the column with the model and adds the missing relationship
constraints.

NOTE: this migration assumes `document_chunks` is empty. Converting a column
that already contains vectors of a different size would fail.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "2f8c1a4b9d10"
down_revision: Union[str, Sequence[str], None] = "69aaa3efd2c7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE document_chunks "
        "ALTER COLUMN embedding TYPE vector(3072)"
    )

    op.create_index(
        "ix_document_chunks_document_id",
        "document_chunks",
        ["document_id"],
    )

    op.create_foreign_key(
        "fk_document_chunks_document_id",
        "document_chunks",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_document_chunks_document_id",
        "document_chunks",
        type_="foreignkey",
    )
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.execute(
        "ALTER TABLE document_chunks "
        "ALTER COLUMN embedding TYPE vector(768)"
    )
