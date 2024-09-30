from response_generator import chatbot_response

def chat():
    print("Start chatting with the bot (type 'quit' to stop)!")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            break
        response = chatbot_response(msg)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
