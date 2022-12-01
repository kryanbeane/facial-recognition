import eventlet
from main import app
from waitress import serve
import socket

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)

if __name__ == '__main__':
    serve(app, host=IPAddr, port=8080, url_scheme='RTMP', threads=6)
