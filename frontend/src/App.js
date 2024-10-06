import React, { useState } from "react";

const App = () => {
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([]);

  const handleSendMessage = async () => {
    if (message.trim()) {
      // Add user's message to the chat history
      setChatHistory([...chatHistory, { sender: "user", text: message }]);

      const res = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message }),
      });

      const data = await res.json();
      // Add bot's response to the chat history
      setChatHistory([
        ...chatHistory,
        { sender: "user", text: message },
        { sender: "bot", text: data.response },
      ]);

      // Clear the input field
      setMessage("");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="bg-white w-full max-w-md shadow-lg rounded-lg p-6">
        <h1 className="text-2xl font-bold mb-4 text-center text-gray-700">
          AI Chatbot
        </h1>

        {/* Chat History */}
        <div className="h-96 overflow-y-auto p-4 border border-gray-300 rounded-lg mb-4">
          {chatHistory.map((chat, index) => (
            <div
              key={index}
              className={`mb-2 p-3 rounded-lg ${
                chat.sender === "user"
                  ? "bg-blue-100 text-right"
                  : "bg-gray-200 text-left"
              }`}
            >
              <p
                className={`${
                  chat.sender === "user" ? "text-blue-700" : "text-gray-700"
                }`}
              >
                {chat.text}
              </p>
            </div>
          ))}
        </div>

        {/* Input and Send Button */}
        <div className="flex items-center space-x-3">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type your message..."
            className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
          <button
            onClick={handleSendMessage}
            className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-5 rounded-lg shadow-lg transition duration-200 ease-in-out"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default App;
