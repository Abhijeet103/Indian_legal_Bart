from transformers import pipeline

def break_lines(text, max_tokens):
    words = text.split()
    lines = [' '.join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]
    return '\n'.join(lines)

def split_text_into_chunks(text, max_tokens):
    words = text.split(" ")
    count = 0
    ans = []
    str = ""
    for word in words :
        str +=word
        count +=1
        if count == max_tokens :
            ans.append(str)
            count =0
            str =""

    return ans



pipe = pipeline('summarization', model='bart_Legal_india_model')
gen_kwargs = {'length_penalty': 0.1, 'num_beams': 8, "max_length": 1024}

input_file_path = 'input.txt'

with open(input_file_path, 'r') as file:
    custom_dialogue = file.read()

# Maximum tokens per string
max_tokens_per_string = 1024


# Split the custom dialogue into chunks
# dialogue_chunks = split_text_into_chunks(custom_dialogue, max_tokens_per_string)
# print(len(dialogue_chunks))
summary = pipe(custom_dialogue, **gen_kwargs)

print(summary)
output_file_path = 'output.txt'


summary = summary[0]['summary_text']
summary = break_lines(summary , 18)

with open(output_file_path, 'w') as output_file:
    output_file.write(summary)
