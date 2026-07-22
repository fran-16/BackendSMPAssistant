import os
import re
import uuid
from datetime import datetime, timezone
from typing import Literal

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter(
    prefix="/storage",
    tags=["storage"],
)

AWS_REGION = os.getenv("AWS_REGION")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")

if not AWS_REGION:
    raise RuntimeError("Falta AWS_REGION")

if not AWS_S3_BUCKET:
    raise RuntimeError("Falta AWS_S3_BUCKET")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    config=Config(signature_version="s3v4"),
)


class MealImageUploadRequest(BaseModel):
    user_id: str
    content_type: Literal[
        "image/jpeg",
        "image/png",
        "image/webp",
    ]


class MealImageUploadResponse(BaseModel):
    upload_url: str
    object_key: str
    expires_in: int


@router.post(
    "/meal-image-upload-url",
    response_model=MealImageUploadResponse,
)
async def create_meal_image_upload_url(
    body: MealImageUploadRequest,
) -> MealImageUploadResponse:

    user_id = body.user_id.strip()

    if not re.fullmatch(r"[A-Za-z0-9_-]{5,128}", user_id):
        raise HTTPException(
            status_code=400,
            detail="UID inválido",
        )

    extension_by_type = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }

    extension = extension_by_type[body.content_type]
    now = datetime.now(timezone.utc)
    image_id = uuid.uuid4().hex

    object_key = (
        f"users/{user_id}/meal-images/"
        f"{now:%Y/%m/%d}/{image_id}.{extension}"
    )

    expires_in = 300

    try:
        upload_url = s3_client.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": AWS_S3_BUCKET,
                "Key": object_key,
                "ContentType": body.content_type,
            },
            ExpiresIn=expires_in,
            HttpMethod="PUT",
        )

        return MealImageUploadResponse(
            upload_url=upload_url,
            object_key=object_key,
            expires_in=expires_in,
        )

    except (BotoCoreError, ClientError) as error:
        raise HTTPException(
            status_code=500,
            detail="No se pudo preparar la carga de la imagen",
        ) from error
