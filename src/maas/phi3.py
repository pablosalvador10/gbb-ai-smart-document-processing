import json
import logging
import os
import ssl
import urllib.request
from typing import Any, Dict

# Configure logging
from utils.ml_logging import get_logger

logger = get_logger()


def allow_self_signed_https(allowed: bool) -> None:
    """
    Bypass the server certificate verification on the client side if allowed.

    Parameters:
    allowed (bool): If True, allows self-signed HTTPS certificates.
    """
    if (
        allowed
        and not os.environ.get("PYTHONHTTPSVERIFY", "")
        and getattr(ssl, "_create_unverified_context", None)
    ):
        ssl._create_default_https_context = ssl._create_unverified_context


def call_phi_3_medium_128k(
    prompt: str,
    system_message: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 1.0,
    stream: bool = False,
) -> Any:
    """
    Calls the Phi-3-medium-128k model with the provided prompt and system message.

    Parameters:
    api_key (str): The API key for the endpoint.
    prompt (str): The user prompt to send to the model.
    system_message (str): The system message to send to the model.
    max_tokens (int): The maximum number of tokens to generate. Default is 1024.
    temperature (float): The sampling temperature. Default is 0.7.
    top_p (float): The nucleus sampling probability. Default is 1.0.
    stream (bool): Whether to stream the response. Default is False.

    Returns:
    Any: The response from the model.
    """
    allow_self_signed_https(True)

    data = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }

    body = str.encode(json.dumps(data))

    url = os.getenv("AZUREAI_ENDPOINT_URL_PHI_3_MEDIUM_128")
    api_key = os.getenv("AZUREAI_ENDPOINT_KEY_PHI_3_MEDIUM_128")
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + api_key}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        return json.loads(result)
    except urllib.error.HTTPError as error:
        logger.error("The request failed with status code: %s", str(error.code))
        logger.error(error.info())
        logger.error(error.read().decode("utf8", "ignore"))
        return None
