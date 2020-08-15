import socket
import numpy as np


class Server:
    def __init__(self, port=12345, max_clients=1):
        s = socket.socket()
        self.host = socket.gethostname()
        self.port = port
        s.bind((self.host, port))
        print('binding to', port,self.host)
        s.listen(max_clients)
        self.clients = []
        for _ in range(max_clients):
            c, addr = s.accept()
            self.clients.append((c, addr))
            print('connection established to', addr)
        self.socket = s

    def __del__(self):
        for c, addr in self.clients:
            c.close()

        self.socket.close()

    def recv(self, client_index, byte_num, chunk_size=2024):
        client, _ = self.clients[client_index]
        return list(sock_recv(client, byte_num, chunk_size))

    def recv_int(self, client_index):
        client, _ = self.clients[client_index]
        return int.from_bytes(sock_recv(client, 4), 'big')

    def send_all(self, data, byte_num):
        returns = []
        for c, _ in self.clients:
            returns.append(sock_send(c, bytes(list(data)),byte_num))
        return returns

    def send_all_int(self, number):
        self.send_all(number.to_bytes(4, 'big'), 4)


class Client:
    def __init__(self, hostname, port=12345,):
        s = socket.socket()
        try:
            s.connect((hostname, port))
            self.socket = s
            print("connected to server")
        except Exception:
            print('could not connect to host')
            self.socket = None

    def __del__(self):
        if self.socket is not None:
            self.socket.close()

    def send(self, data, byte_num):
        if self.socket is not None:
            sock_send(self.socket, bytes(list(data)),byte_num)

    def send_int(self, number):
        if self.socket is not None:
            sock_send(self.socket, number.to_bytes(4, 'big'), 4)

    def recv(self, byte_num, chunk_size=2024):
        if self.socket is not None:
            return list(sock_recv(self.socket, byte_num, chunk_size))
        return 0

    def recv_int(self):
        if self.socket is not None:
            return int.from_bytes(sock_recv(self.socket, 4), 'big')
        return 0


def sock_send(socket, data, byte_num):
    progress = 0
    while progress < byte_num:
        sent = socket.send(data[progress:])
        if sent == 0:
            raise RuntimeError("socket connection broken")
        progress += sent


def sock_recv(socket, byte_num, chunk_size=2024):
    received_data = []
    bytes_received = 0
    while bytes_received < byte_num:
        chunk = socket.recv(min(byte_num-bytes_received, chunk_size))
        received_data.append(chunk)
        bytes_received += len(chunk)
    return b''.join(received_data)


if __name__ == '__main__':
    server = Server()
    height = server.recv_int(0)
    width = server.recv_int(0)
    image = server.recv(0, height*width*3)
    print(height, width, np.array(image).reshape((height, width, 3)))

