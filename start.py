# from flask import Flask, jsonify, request
# import subprocess
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)
# # Function to start the chatbot server
# def start_chatbot_server():
#     # Execute the shell commands to start the chatbot server
#     # subprocess.run(['conda', 'activate', 'pdfchat'])
#     subprocess.run(['ollama', 'pull', 'nomic-embed-text'])
#     subprocess.run(['chainlit', 'run', 'app.py'])

# # Route to handle requests to the chatbot endpoint
# @app.route('/chatbot', methods=['GET'])
# def chatbot():
#     if request.method == 'GET':
#         return jsonify({'response': 'PDF chatbot server is running'})

# # Start the chatbot server when the Flask app starts up
# if __name__ == '__main__':
#     start_chatbot_server()
#     app.run(debug=True, port=5000)
