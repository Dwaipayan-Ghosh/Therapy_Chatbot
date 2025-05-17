from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "victunes/TherapyLlama-8B-v1"
AutoModelForCausalLM.from_pretrained(model_name).save_pretrained("./local_model")
AutoTokenizer.from_pretrained(model_name).save_pretrained("./local_model")