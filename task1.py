responses = {
    'hello': 'Hi there! How can I help you?',
    'how are you': 'I am functioning at 100%, thanks!',
    'what is ai': 'AI is the simulation of human intelligence by machines.',
    'bye': 'Goodbye! Have a great day!',
    'help': 'I can answer questions about AI. Try asking me something!'
}

print("DecodeLabs Chatbot — type 'exit' to quit")

while True:
    raw_input_text = input('You: ')
    clean_input = raw_input_text.lower().strip()
    
    if clean_input in ('exit', 'quit'):
        print('Bot: Shutting down. Goodbye!')
        break
    
    reply = responses.get(clean_input, 'I do not understand. Try rephrasing!')
    print(f'Bot: {reply}')