#include "Matrix.h"

int main()
{
    int serverSocket, clientSocket;            // 服务端和客户端套接字(文件描述符)
    struct sockaddr_in serverAddr, clientAddr; // 服务器和客户端地址结构
    socklen_t clientAddrLen;                   // 客户端地址长度
    float buffer[BUFFER_SIZE];

    // 创建服务端套接字
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1)
    {
        perror("Failed to create socket");
        exit(EXIT_FAILURE);
    }

    // 设置服务器地址信息
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(SERVER_PORT);
    serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);

    // 绑定套接字到指定地址和端口
    if (bind(serverSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) == -1)
    {
        perror("Failed to bind socket");
        exit(EXIT_FAILURE);
    }

    // 监听连接请求
    if (listen(serverSocket, 5) == -1)
    {
        perror("Failed to listen");
        exit(EXIT_FAILURE);
    }

    while (true)
    {
        // 接受客户端连接
        clientAddrLen = sizeof(clientAddr);
        clientSocket = accept(serverSocket, (struct sockaddr *)&clientAddr, &clientAddrLen);
        if (clientSocket == -1)
        {
            perror("Failed to accept client connection");
            exit(EXIT_FAILURE);
        }

        // 接收数据

        memset(buffer, 0, BUFFER_SIZE * sizeof(float));                                // 清空缓冲区
        int receivedSize = recv(clientSocket, buffer, BUFFER_SIZE * sizeof(float), 0); // 接收数据
        if (receivedSize == -1)
        {
            perror("Failed to receive data");
            exit(EXIT_FAILURE);
        }
        else
        {
            int numFloats = receivedSize / sizeof(float); // 计算接收到的浮点数数量

            if (numFloats == 784)
            {
                printf("Received %d floats from client.\n", numFloats);
                // 处理数据
                Matrix<float> input(1, 784);
                for (int i = 0; i < 784; ++i)
                {
                    input(0, i) = buffer[i];
                }
                Matrix<float> output = model<float>("/home/wmx/桌面/project/GKDproject/project/mnist-fc")._predict(input); // 预测

                // 发送响应给客户端
                memset(buffer, 0, BUFFER_SIZE * sizeof(float)); // 清空缓冲区
                for (int i = 0; i < 10; ++i)
                {
                    buffer[i] = output(0, i);
                }
                if (send(clientSocket, buffer, 10 * sizeof(float), 0) == -1)
                {
                    perror("Failed to send response");
                    exit(EXIT_FAILURE);
                }
            }
            else
            {
                // printf("Error: Expected 784 floats, but received %d floats.\n", numFloats);
            }
        }
    }
    // 关闭套接字
    close(clientSocket);
    close(serverSocket);

    return 0;
}
