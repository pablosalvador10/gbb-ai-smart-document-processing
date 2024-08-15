import base64
import json
import logging
import os
import ssl
import urllib.request
from typing import Optional

# Configure logging
from utils.ml_logging import get_logger

logger = get_logger()


def allow_self_signed_https(allowed):
    """
    Bypass the server certificate verification on the client side.

    Parameters:
    allowed (bool): If True, allows self-signed HTTPS certificates.
    """
    if (
        allowed
        and not os.environ.get("PYTHONHTTPSVERIFY", "")
        and getattr(ssl, "_create_unverified_context", None)
    ):
        ssl._create_default_https_context = ssl._create_unverified_context


def phi_3_vision_inference(prompt: str, image_path: str) -> Optional[dict]:
    """
    Classify an image using the provided classification prompt and return the results as a JSON object.

    Parameters:
    prompt (str): The prompt to guide the classification.
    image_path (str): The file path to the image to be classified.

    Returns:
    Optional[dict]: The classification result from the API as a JSON object, or None if the request fails.
    """
    allow_self_signed_https(True)

    # Encode the image
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Create the user message content
    user_message_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}",
            },
        },
        {
            "type": "text",
            "text": prompt,
        },
    ]

    # Create the data dictionary
    data = {
        "input_data": {
            "input_string": [
                {
                    "role": "user",
                    "content": user_message_content,
                }
            ],
            "parameters": {"temperature": 0, "max_new_tokens": 2048},
        }
    }

    body = str.encode(json.dumps(data))

    url = os.environ.get("AZUREAI_ENDPOINT_URL_PHI_3_VISION")
    api_key = os.environ.get("AZUREAI_ENDPOINT_KEY_PHI_3_VISION")
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "azureml-model-deployment": "phi-3-vision-128k-instruct-2",
    }

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode("utf-8")

        # Load the result into a JSON object
        result_json = json.loads(result)

        # Remove spaces from the values in the JSON object
        def remove_spaces(obj):
            if isinstance(obj, dict):
                return {k: remove_spaces(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [remove_spaces(i) for i in obj]
            elif isinstance(obj, str):
                return obj.replace(" ", "")
            else:
                return obj

        cleaned_result_json = remove_spaces(result_json)

        return cleaned_result_json
    except urllib.error.HTTPError as error:
        logger.error("The request failed with status code: %s", str(error.code))
        logger.error(error.info())
        logger.error(error.read().decode("utf8", "ignore"))
        return None
