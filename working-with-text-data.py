### TOKENIZING TEXT USING REGULAR EXPRESSIONS ###

# Reading in a short story as text sample into Python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99]) # Passed test'''

# Splitting text into individual words, whitespaces, and punctionation characters
import re
text = "Hello, world! This is a test!"
result = re.split(r'(\s)', text)
#print(result) # Passed test

# Modifying the above to include commas, question marks, periods, and exclamation points
result = re.split(r'([,.!?]|\s)', text)
#print (result) # passed test

# Further modifying to remove redundant characters such as empty strings and whitespaces
result = [item for item in result if item.strip()]
#print(result) # Passed test

# Further modifying to include more punctionation marks; finalized tokenizer
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
#print(result) # Passed test

# Test for tokenizer on the short story text
preprocessed = re.split(r'([,.:?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed)) # Passed test
print(preprocessed[:30]) # Passed test

### CONVERTING TOKENS INTO TOKEN IDS ###

# Creating a vocabulary of unique tokens from the preprocessed text
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("Vocabulary size:", vocab_size) # Passed test

vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    #print(item)
    if i >= 50:
        break
        # Passed test

# Implementing a simple text tokenizer function
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids) # Passed test
print(tokenizer.decode(ids)) # Passed test

### ADDING SPECIAL CONTEXT TOKENS ###

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items())) # Passed test

for i, item in enumerate(list(vocab.items()) [-5:]):
    print(item) # Passed test

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text) # Passed test

print(tokenizer.encode(text)) # Passed test
print(tokenizer.decode(tokenizer.encode(text))) # Passed test

### BYTE PAIR ENCODING ###

from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers) # Passed test

strings = tokenizer.decode(integers)

print(strings)  # Passed test

### DATA SAMPLING WITH A SLIDING WINDOW ###

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text)) # Passed test

enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}") # Passed test
print(f"y:      {y}") # Passed test

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired) # Passed test

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired])) # Passed test

