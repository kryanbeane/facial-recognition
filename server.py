import eventlet
import socketio

from main import app
from waitress import serve
import socket


sio = socketio.Server()
appServer = socketio.WSGIApp(sio, app)
hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)

if __name__ == '__main__':
    serve(appServer, host=IPAddr, port=8080, url_scheme='http', threads=6, log_untrusted_proxy_headers=True)
