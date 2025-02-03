# SambaNova Data Analysis Assistant

## Overview

This Streamlit application serves as a **Mid-Senior Level Data Analysis Agent** powered by the **SambaNova API**. It allows users to upload datasets, perform data analysis, and interact with the data using natural language queries. The application leverages **PandasAI** and **Langchain** to provide intelligent data analysis capabilities.

## Features

- **Dark Theme**: A sleek, dark-themed UI for a modern and comfortable user experience.
- **API Authentication**: Securely authenticate using your SambaNova API key.
- **Data Upload**: Upload CSV or Excel files for analysis.
- **Data Preview**: Preview uploaded datasets and view general information such as shape, data types, and memory usage.
- **Data Merging**: Merge two datasets based on common columns.
- **Natural Language Querying**: Ask questions about your data in natural language and get intelligent responses.
- **Response Time**: Display the time taken to process each query.

## Setup

### Prerequisites

- **SambaNova API Key**: Obtain your API key from [SambaNova](https://www.sambanova.ai/).

### Installation

1. **Install Required Libraries**:
   ```bash
   pip install streamlit pandas pandasai langchain-openai
