# Bilingual Q&A System

This project is a Bilingual Question and Answer System that uses a fine-tuned language model to answer questions in English and Bangla. It leverages the Hugging Face Transformers library and Gradio for creating a user-friendly interface.

## Features

- Fine-tune a pre-trained language model for Q&A tasks.
- Supports questions in both English and Bangla.
- Provides a web-based interface using Gradio.

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Gradio
- Datasets

## Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install the required packages:**

   ```bash
   pip install torch transformers gradio datasets
   ```

3. **Set the Hugging Face API token:**

   Ensure that the `HUGGINGFACE_HUB_TOKEN` environment variable is set. You can set it in your terminal or command prompt:

   ```bash
   export HUGGINGFACE_HUB_TOKEN=your_huggingface_token
   ```

   Or, on Windows:

   ```cmd
   set HUGGINGFACE_HUB_TOKEN=your_huggingface_token
   ```

## Usage

1. **Prepare your data:**

   Ensure you have a JSON file named `my_data.json` with the following structure:

   ```json
   [
       {
           "instruction": "Your instruction here",
           "input": "Your input here",
           "output": "Expected output here"
       },
       ...
   ]
   ```

2. **Run the main script:**

   ```bash
   python QnAsystem.py
   ```

   This will load the data, fine-tune the model, and launch the Gradio interface.

3. **Access the Gradio interface:**

   Open the provided URL in your browser to interact with the Q&A system.

## Security Note

- **Do not commit API tokens or secrets directly in your code.** Use environment variables or configuration files to manage sensitive information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Gradio](https://github.com/gradio-app/gradio)
