def get_classification_prompt(result_ocr):
    """
    Generate the document classification prompt with the given OCR result.

    Parameters:
    result_ocr (str): The scanned text from the OCR of the document.

    Returns:
    str: The formatted classification prompt.
    """
    CLASSIFICATION_PROMPT = f"""
    ### Task: Document Classification

    #### Inputs:
    - **TEXT**: Scanned text from the OCR of the document to be classified. The input document follows the Markdown format.

    {result_ocr}

    #### Instructions:
    You are required to classify the document based on the provided scanned text. The document should be categorized into the most appropriate category from the list below. If the document does not fit neatly into one category, choose the category that best matches the majority of its characteristics.

    1. **letter**:
        - **Description**: A written or printed message addressed to a specific person or organization.
        - **Common Characteristics**:
            - Contains salutations and closings.
            - Often addressed to a specific person or entity.
            - Includes date, sender's and recipient's addresses.

    2. **form**:
        - **Description**: A document with blank fields for the user to fill out with specific information.
        - **Common Characteristics**:
            - Contains predefined fields and labels.
            - Includes spaces for user input.
            - Often used for data collection, applications, and surveys.

    3. **email**:
        - **Description**: An electronic message exchanged between people using electronic devices.
        - **Common Characteristics**:
            - Contains email headers such as "From", "To", "Subject", and "Date".
            - Includes conversational text.
            - May include attachments or references to other documents.

    4. **handwritten**:
        - **Description**: Any document written manually by hand.
        - **Common Characteristics**:
            - Contains handwritten text.
            - May include various writing styles and penmanship.
            - Often lacks formal structure.

    5. **advertisement**:
        - **Description**: A public notice promoting a product, service, or event.
        - **Common Characteristics**:
            - Contains promotional language and visuals.
            - Includes information about a product, service, or event.
            - Often designed to attract attention.

    6. **scientific report**:
        - **Description**: A detailed account of a scientific study or experiment.
        - **Common Characteristics**:
            - Includes sections such as introduction, methods, results, and conclusion.
            - Contains data, graphs, and references.
            - Written in a formal and structured manner.

    7. **scientific publication**:
        - **Description**: An article published in a scientific journal.
        - **Common Characteristics**:
            - Includes abstract, introduction, methodology, results, and discussion.
            - Contains citations and references.
            - Peer-reviewed and follows academic standards.

    8. **specification**:
        - **Description**: A detailed description of the requirements, design, or performance of a product or system.
        - **Common Characteristics**:
            - Includes technical details and standards.
            - Structured format with headings and subheadings.
            - Often used in engineering and manufacturing.

    9. **file folder**:
        - **Description**: A document that serves as a cover or holder for other documents.
        - **Common Characteristics**:
            - Contains a label or title indicating the contents.
            - Often used for organizing and storing multiple related documents.
            - May include a table of contents or summary.

    10. **news article**:
        - **Description**: A written piece reporting on current events or topics of interest.
        - **Common Characteristics**:
            - Contains headlines and bylines.
            - Includes factual reporting and quotes.
            - Published in newspapers, magazines, or online platforms.

    11. **budget**:
        - **Description**: A financial plan outlining expected income and expenses.
        - **Common Characteristics**:
            - Includes numerical data and tables.
            - Details various categories of income and expenditure.
            - Often used for financial planning and analysis.

    12. **invoice**:
        - **Description**: A commercial document issued by a seller to a buyer, indicating the products, quantities, and agreed prices for products or services.
        - **Common Characteristics**:
            - Contains the term "Invoice".
            - Includes seller and buyer information, invoice number, date, and payment terms.
            - Lists products or services provided, quantities, unit prices, and total amount due.
            - May include tax details and payment instructions.

    13. **presentation**:
        - **Description**: A document used to communicate information visually, often as slides.
        - **Common Characteristics**:
            - Includes slides with text, images, and graphs.
            - Organized in a structured format.
            - Often used in meetings and lectures.

    14. **questionnaire**:
        - **Description**: A set of questions designed to gather information from respondents.
        - **Common Characteristics**:
            - Contains multiple questions.
            - May include multiple-choice, open-ended, or scale-based questions.
            - Used for surveys and research.

    15. **resume**:
        - **Description**: A document summarizing an individual's education, work experience, and skills.
        - **Common Characteristics**:
            - Includes sections such as contact information, work experience, education, and skills.
            - Written in a concise and structured format.
            - Used for job applications.

    16. **memo**:
        - **Description**: A brief written message used for internal communication within an organization.
        - **Common Characteristics**:
            - Contains a header with the recipient, sender, date, and subject.
            - Includes concise information or instructions.
            - Often used for announcements and updates.

    #### Steps to Classify the Document:
    1. Carefully examine the text provided in markdown format, which includes the OCR output of the scanned document.
    2. Identify key elements and details within the document that match the descriptions of the categories listed above.
    3. Select the category that best fits the document based on its content and purpose. If the document seems to fit into more than one category, choose the one that matches the majority of its characteristics.
    4. Provide the name of the category as a single word (e.g., letter, form, email) based on your analysis of the document content.

    #### Output:
    - Return only the chosen category as a plain string without any additional characters. Remember the categories are: "letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication", "specification", "file folder", "news article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo" Do not add any additional characters."""
    return CLASSIFICATION_PROMPT
