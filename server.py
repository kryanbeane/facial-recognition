from main import app
import waitress
import socket

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)

if __name__ == '__main__':
    waitress.serve(app, host=IPAddr, port=8080, url_scheme='HTTP', threads=5)
