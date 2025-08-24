
#These are the Import Statements
from transformers import T5Tokenizer, T5ForConditionalGeneration

text = """
He stepped away from the mic. This was the best take he
 had done so far, but something seemed missing. Then it struck him all at once. 
 Visuals ran in front of his eyes and music rang in his ears. His eager fingers went to
  work in an attempt to capture his thoughts hoping the results would produce something that was
   at least half their glory.
"""

print(f"The length of the text before summarization is : {len(text)} ")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_text = "summarize : " + text

#ensures the max length is 512 and if it goes beyond that it will cut it off
inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)


summary_idices = model.generate(inputs, max_length=40, min_length=10,length_penalty = 2.0, num_beams=4, early_stopping=True)

summary = tokenizer.decode(summary_idices[0], skip_special_tokens=True)


print(f"The length of the text after summarization is : {len(summary)} ")
print(summary)
