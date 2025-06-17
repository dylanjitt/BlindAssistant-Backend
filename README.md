you need Python 3
and huggingface account with acces to the following models:
-ollama
-ngrok

steps
1. git clone
2. python3 -m venv venv
3. source venv/bin/activate
4. pip install -r requiremets.txt
5. huggingface-cli login 
(you should get  your api key with access to the models)
6. fastapi dev src/api.py