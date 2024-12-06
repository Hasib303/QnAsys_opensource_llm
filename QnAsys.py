import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import gradio as gr
from datasets import Dataset
import json
import os

class QnASystem:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        # Set the Hugging Face token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
    def load_data(self, json_file="my_data.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        train_data = {
            "instruction": [item["instruction"] for item in data],
            "input": [item["input"] for item in data],
            "output": [item["output"] for item in data]
        }
        return Dataset.from_dict(train_data)

    def preprocess_function(self, examples):
        prompts = [
            f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
            for instruction, input_text, output in zip(
                examples["instruction"], 
                examples["input"], 
                examples["output"]
            )
        ]
        
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        return tokenized

    def train(self, dataset, output_dir="qa_model"):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=100,
            save_total_limit=2,
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=lambda data: {'input_ids': torch.stack([f["input_ids"] for f in data]),
                                      'attention_mask': torch.stack([f["attention_mask"] for f in data])}
        )
        
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def generate_answer(self, question, max_length=100):
        inputs = self.tokenizer(
            f"Instruction: {question}\nInput: \nOutput:",
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def create_gradio_interface(self):
        def answer_question(question):
            return self.generate_answer(question)
        
        iface = gr.Interface(
            fn=answer_question,
            inputs=gr.Textbox(label="Enter your question (English or Bangla):"),
            outputs=gr.Textbox(label="Answer"),
            title="Bilingual Q&A System",
            description="Ask questions in English or Bangla"
        )
        return iface

def main():
    qa_system = QnASystem()
    
    # Load and preprocess the training data
    dataset = qa_system.load_data()
    
    # Fine-tune the model
    qa_system.train(dataset)
    
    # Create and launch the Gradio interface
    interface = qa_system.create_gradio_interface()
    interface.launch()

if __name__ == "__main__":
    main()
