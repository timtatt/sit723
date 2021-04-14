from chatterbot import ChatBot

bot = ChatBot(
    'Steve',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///database.sqlite3',
    logic_adapters=[
        'chatterbot.logic.BestMatch'
    ])