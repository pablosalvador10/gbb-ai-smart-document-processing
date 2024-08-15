# 🚀 Advanced OCR, Intelligent Tagging, and NER with Azure AI 

Azure AI offers advanced capabilities for document classification, enabling enterprises to automate and streamline document processing workflows. This project leverages Azure AI Document Intelligence, Large Language Models (LLMs), and Small Language Models (SLMs) to accurately classify and extract information from commercial documents like invoices, receipts, and contracts. The result is searchable, structured data from previously unstructured sources, enhanced by semantic natural language understanding with a vector-based approach.

We will explore the latest technologies and approaches, helping you make informed decisions to leverage advanced Optical Character Recognition (OCR), Named Entity Recognition (NER), summarization, vectorization, and indexing to make your data searchable.

For more context and a detailed explanation, please refer to the full blog post. [here]()

## 🧭 Guide to Decision-Making 

Choosing the right technology is crucial in engineering decision-making. Refer to the mind map below for guidance on aligning your needs with the best approach.

![Mind Map](utils\images\image.png)

Below are the approaches to various technologies and methods, each linked to a corresponding notebook for detailed code examples and evaluation methodology.

1. **Define Your Success Thresholds**

    a. **Evaluation Criteria**: Learn how to choose the right methodology to compare multiple approaches using quality metrics like accuracy. For a multiclass problem like our use case, see the code [here](01-build-evaluation-methodology.ipynb).

2. **Document Classification (OCR)**

    a. **OCR + LLM**: First, scan the document using OCR, then pass it to an LLM/SLM for extracting the targeted information. See the code [here](04-classification-document-ocr-llm.ipynb).

    b. **Leveraging Multimodality**: Utilize advanced multimodal models like GPT-4 Omni or Phi-3 Vision that can directly accept images for classification. See the code [here](02-classification-document-llm-slm-multimodal.ipynb).

    c. **Fine-Tuning Neural Document Intelligence Models**: Fine-tune pre-trained Azure AI Document Intelligence models with your own data for improved accuracy. See the code [here](03-classification-custom-document-intelligence.ipynb).

3. **Extracting Content**

    a. **Leveraging Multimodality**: Utilize advanced multimodal models like GPT-4 Omni to extract content and summarize per document. See the code [here](05-entity-extraction-document-intelligence.ipynb).

4. **Make Your Data Searchable**

    a. **Vectorization and Indexing**: Use Azure OpenAI to vectorize and the Push SDK to index documents into Azure AI Search, enabling state-of-the-art retrieval approaches and advanced search capabilities. See the code [here](06-make-your-data-searchable.ipynb).

## 📂 Case Study: Making Your Enterprise Unlabeled Document Archives Searchable

### Problem

Enterprises often have millions of unprocessed documents in their archives, leading to inefficiencies. Our goal is to categorize these documents into 16 initial categories, enabling chat functionality, key information extraction, and data understanding. We aim to make this data searchable using Azure AI Search with high relevance scores.

### Data

We use the RVL-CDIP dataset, consisting of 400,000 grayscale images in 16 classes. For this prototype, we selected 100 samples per class, split into 70% for training and 30% for validation.For more information about the RVL-CDIP dataset, please refer to the (dataset page)[https://huggingface.co/datasets/aharley/rvl_cdip] on Hugging Face 📚

### Solution

![Pipeline Diagram](utils/images/image-1.png)

1. **Document Classification**: 
   - We classify documents into 16 categories by fine-tuning pre-trained neural models using Azure AI Document Intelligence.
   - See the code [here](03-classification-custom-document-intelligence.ipynb).

2. **Key Elements Extraction and Summarization**: 
   - Extract and summarize key elements (NER) from documents classified as invoices using language models with multimodality capability for contextual entity extraction and summarization.
   - Enforce proper validation and convert them into a structured format (JSON) with the support of the pydantic and instructor libraries.
   - See the code [here](05-entity-extraction-document-intelligence.ipynb).

3. **Data Indexing and Vectorization**: 
   - Index and vectorize the JSON containing the key information and summarization per document into Azure AI Search.
   - This allows your enterprise to query the documents in a "Bing-like" manner and make your previously unlabeled data searchable.
   - See the code [here](06-make-your-data-searchable.ipynb).


## 📚 Resources

- **Document Intelligence**: For detailed information on Document Intelligence AI and its components, visit our [Documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/?view=doc-intel-4.0.0).
- **Azure AI Studio**: Check out our [Tutorials](https://learn.microsoft.com/en-us/azure/ai-studio/what-is-ai-studio) for hands-on guides on Azure AI Studio.
- **Azure AI Agentic Frameworks**: Explore the [Azure AI Agentic Frameworks Repository](https://github.com/pablosalvador10/gbbai-azure-ai-agentic-frameworks) for frameworks and tools to build AI agents.
- **LLM/SLM Evaluation Framework**: Check out the [LLM/SLM Evaluation Framework Repository](https://github.com/pablosalvador10/gbb-ai-llm-slm-evaluation-framework) for evaluating large and small language models.

## Contributing

We welcome contributions to enhance the capabilities and features of this project. Please read our [contributing guidelines](CONTRIBUTING.md) for more information.

### Disclaimer
> [!IMPORTANT]
> This software is provided for demonstration purposes only. It is not intended to be relied upon for any purpose. The creators of this software make no representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability, suitability or availability with respect to the software or the information, products, services, or related graphics contained in the software for any purpose. Any reliance you place on such information is therefore strictly at your own risk.