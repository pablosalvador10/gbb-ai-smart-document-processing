import base64
import os
from pydantic import BaseModel, Field, ValidationError, root_validator
from typing import Optional, List, Any, Dict
import instructor
from openai import AzureOpenAI
import json
from utils.ml_logging import get_logger

# Initialize logging
logger = get_logger()

def initialize_client():
    """
    Initialize the AzureOpenAI client.

    Returns:
        instructor.Client: The initialized AzureOpenAI client.

    Raises:
        EnvironmentError: If the required environment variables are not set.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("DEPLOYMENT_VERSION")
    api_key = os.getenv("OPENAI_API_KEY")

    if not all([azure_endpoint, api_version, api_key]):
        raise EnvironmentError("Azure OpenAI environment variables are not set properly.")

    return instructor.from_openai(AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        api_key=api_key,
    ))

client = initialize_client()

class Item(BaseModel):
    """
    Represents an item in the invoice.
    """
    name: str
    price: float
    quantity: int

class Invoice(BaseModel):
    """
    Represents an invoice with multiple items and additional details.
    """
    items: List[Item]
    total: float
    reference_number: str
    signature_on_document: str
    origin_address: str
    destination_address: str
    summary: Optional[str] = None 

    @root_validator(pre=True)
    def check_total_and_reference_number(cls, values: dict) -> dict:
        """
        Validates the total amount and reference number in the invoice.

        Args:
            values (dict): The values to validate.

        Returns:
            dict: The validated values.

        Raises:
            ValueError: If the reference number is missing.
        """
        items = values.get('items', [])
        total = values.get('total')
        reference_number = values.get('reference_number')
        
             # Check if the total matches the sum of item prices
        calculated_total = sum(item['price'] * item['quantity'] for item in items)
        if calculated_total != total:
            invoice_id = values.get('reference_number', 'Unknown ID') 
            logger.warning(f"Total amount mismatch for Invoice ID {invoice_id}: Calculated total is {calculated_total}, but the provided total is {total}")
        
        if not reference_number:
            raise ValueError("Reference number is missing")
        
        return values

def extract_invoice(file_path: str) -> Invoice:
    """
    Extracts details from an invoice image and returns an Invoice object.

    Args:
        file_path (str): The path to the invoice image file.

    Returns:
        Invoice: The extracted invoice details.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If an error occurs while reading the file or communicating with the Azure OpenAI service.
        ValueError: If there is an error in parsing the receipt details.
    """
    try:
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")

    # Encode the image in base64
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    # Create the user message with the encoded image
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                },
            },
            {
                "type": "text",
                "text": (
                    "You will be analyzing an image of an invoice and extracting specific details from it. "
                    "When I provide an image, all further input from the 'Instructor:' will be related to extracting information from that image.\n\n"
                    "## Details to Extract:\n"
                    "1. **Items**: For each item listed on the invoice, extract the following:\n"
                    "   - **Name**: The name or description of the item.\n"
                    "   - **Price**: The price of the item.\n"
                    "   - **Quantity**: The quantity of the item.\n\n"
                    "2. **Total Amount**: Extract the total amount listed on the invoice.\n\n"
                    "3. **Signature**: Extract any signature present on the invoice. If no signature is present, indicate 'No signature present'.\n\n"
                    "4. **Origin Address**: Extract the origin address from the invoice.\n\n"
                    "5. **Destination Address**: Extract the destination address from the invoice.\n\n"
                    "6. **Reference Number**: Extract the reference number from the invoice. Ensure the reference number is correctly identified and extracted.\n\n"
                    "7. **Summary**: Provide a summary of the document, describing in detail what the document is about and some of the key details.\n\n"
                    "## Formatting Guidelines:\n"
                    "Ensure the extracted information is clearly formatted as follows:\n\n"
                    "### Items:\n"
                    "- **Item 1**: Name, Price, Quantity\n"
                    "- **Item 2**: Name, Price, Quantity\n"
                    "- (and so on for each item)\n\n"
                    "### Total Amount:\n"
                    "- **Total Amount**: [Total Amount]\n\n"
                    "### Signature:\n"
                    "- **Signature**: [Signature or 'No signature present']\n\n"
                    "### Origin Address:\n"
                    "- **Origin Address**: [Origin Address]\n\n"
                    "### Destination Address:\n"
                    "- **Destination Address**: [Destination Address]\n\n"
                    "### Reference Number:\n"
                    "- **Reference Number**: [Reference Number]\n\n"
                    "### Summary:\n"
                    "- **Summary**: [Summary]\n\n"
                    "## Accuracy:\n"
                    "Ensure all details are accurate and clearly labeled. Double-check the extracted information to ensure it matches the details in the invoice image. "
                    "Pay special attention to the reference number to ensure it is correctly identified and extracted."
                ),
            },
        ],
    }

    try:
        result = client.chat.completions.create(
            model=os.getenv("DEPLOYMENT_ID"),
            max_tokens=4000,
            response_model=Invoice,
            temperature=0,
            messages=[user_message],
        )
    except Exception as e:
        raise Exception(f"An error occurred while communicating with the Azure OpenAI service: {e}")

    # Parse the result and return as Invoice
    if result:
        try:
            receipt_data = result.model_dump()
            receipt = Invoice(**receipt_data)
            return receipt
        except ValidationError as e:
            raise ValueError(f"Error in parsing receipt details: {e}")
    else:
        raise ValueError("No response from the Azure OpenAI service")
    



def invoice_to_json(invoice: Invoice) -> Dict[str, Any]:
    """
    Convert an Invoice instance to a JSON object that matches the combined index fields schema.

    Args:
        invoice (Invoice): The Invoice instance to convert.

    Returns:
        Dict[str, Any]: The JSON object representation of the Invoice.
    """
    invoice_dict = invoice.dict()

    # Create the JSON object matching the combined index fields schema
    json_object = {
        "id": invoice_dict.get("reference_number"),
        "content": invoice_dict.get("summary", ""),  # Assuming content might be part of the invoice
        "content_vector": invoice_dict.get("content_vector", []),  # Assuming content_vector might be part of the invoice
        "total": invoice_dict.get("total"),
        "reference_number": invoice_dict.get("reference_number"),
        "signature_on_document": invoice_dict.get("signature_on_document"),
        "origin_address": invoice_dict.get("origin_address"),
        "destination_address": invoice_dict.get("destination_address"),
        "items_purchased": [
            {"list_item": f"{item.name}, {item.price}, {item.quantity}"}
            for item in invoice.items
        ],
    }

    return json_object
