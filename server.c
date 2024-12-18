/////////////////////////////////////////////////////
// Made by Mikus (Cappuch) - 2024 December 17th    //
//                                                 //
// Simple Web Server (HTTP)                        //
// 200 lines worth of wasted storage and time      //
//                                                 //
// MIT License                                     //
/////////////////////////////////////////////////////

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    #include <direct.h>
    #define mkdir(dir, mode) _mkdir(dir)
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <unistd.h>
    #include <sys/stat.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

#define PORT 8080
#define BUFFER_SIZE 4096 * 4 * 2 // 32KB
#define MAX_PATH_LENGTH 256

void handle_client(int client_socket);
char* serve_file(const char* filepath, const char* content_type, size_t* content_length);
char* create_response(const char* status, const char* content_type, const char* content, size_t content_length);
void route_request(int client_socket, const char* path);

int main() {
    #ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            printf("WSAStartup failed\n"); // Hey..... Ur stuff is not working... Works on my machine11!
            return 1;
        }
    #endif

    struct stat st;
    if (stat("posts", &st) == -1) {
        printf("Creating posts directory\n");
        #ifdef _WIN32
            if (_mkdir("posts") != 0) {
                printf("Error creating directory: %s\n", strerror(errno));
                return 1;
            }
        #else
            if (mkdir("posts", 0700) != 0) { // 0700 = rwx------
                printf("Error creating directory: %s\n", strerror(errno));
                return 1;
            }
        #endif
    }

    int server_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);

    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        printf("Error creating socket\n");
        return 1;
    }

    int opt = 1;
    setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        printf("Error binding socket\n");
        return 1;
    }

    if (listen(server_socket, 1) < 0) {
        printf("Error listening\n");
        return 1;
    }

    printf("Server listening on 0.0.0.0:%d\n", PORT);
    printf("http://localhost:%d\n", PORT);

    while (1) {
        int client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
        if (client_socket < 0) {
            printf("Error accepting connection\n");
            continue;
        }
        handle_client(client_socket);
    }

    return 0;
}

void handle_client(int client_socket) {
    char buffer[BUFFER_SIZE];
    memset(buffer, 0, BUFFER_SIZE);

    ssize_t bytes_received = recv(client_socket, buffer, BUFFER_SIZE - 1, 0);
    if (bytes_received > 0) {
        char method[16], path[MAX_PATH_LENGTH], protocol[16];
        sscanf(buffer, "%s %s %s", method, path, protocol);
        
        route_request(client_socket, path);
        
        printf("Request: %s %s from client\n", method, path);
    }

    #ifdef _WIN32
        closesocket(client_socket);
    #else
        close(client_socket);
    #endif
}

char* serve_file(const char* filepath, const char* content_type, size_t* content_length) {
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        *content_length = 13;
        return strdup("404 Not Found");
    }

    fseek(file, 0, SEEK_END);
    *content_length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* content = malloc(*content_length + 1);
    if (!content) {
        fclose(file);
        return NULL;
    }

    fread(content, 1, *content_length, file);
    content[*content_length] = '\0';
    fclose(file);

    return content;
}

char* create_response(const char* status, const char* content_type, const char* content, size_t content_length) {
    char* response = malloc(BUFFER_SIZE);
    if (!response) return NULL;

    snprintf(response, BUFFER_SIZE,
        "HTTP/1.1 %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "\r\n"
        "%s",
        status, content_type, content_length, content);

    return response;
}

void route_request(int client_socket, const char* path) {
    size_t content_length;
    char* content;
    char* response;
    char filepath[MAX_PATH_LENGTH];
    const char* content_type;

    if (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0) {
        content = serve_file("templates/index.html", "text/html", &content_length);
        response = create_response("200 OK", "text/html", content, content_length);
    }
    else if (strcmp(path, "/") == 0 || strcmp(path, "/script.js") == 0) {
        content = serve_file("src/script.js", "application/javascript", &content_length);
        response = create_response("200 OK", "application/javascript", content, content_length);
    }
    else if (strcmp(path, "/") == 0 || strcmp(path, "/styles.css") == 0) {
        content = serve_file("src/styles.css", "text/css", &content_length);
        response = create_response("200 OK", "text/css", content, content_length);
    }
    else if (strncmp(path, "/static/", 8) == 0) {
        const char* clean_path = path + 8;
        while (strstr(clean_path, "../") != NULL) { // kekw
            clean_path = strstr(clean_path, "../") + 3;
        }

        snprintf(filepath, MAX_PATH_LENGTH, "static/%s", clean_path);

        const char* ext = strrchr(filepath, '.');
        if (ext) {
            if (strcmp(ext, ".html") == 0) content_type = "text/html";
            else if (strcmp(ext, ".css") == 0) content_type = "text/css";
            else if (strcmp(ext, ".js") == 0) content_type = "application/javascript";
            else if (strcmp(ext, ".png") == 0) content_type = "image/png";
            else if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0) content_type = "image/jpeg";
            else if (strcmp(ext, ".gif") == 0) content_type = "image/gif";
            else if (strcmp(ext, ".svg") == 0) content_type = "image/svg+xml";
            else if (strcmp(ext, ".ico") == 0) content_type = "image/x-icon";
            else if (strcmp(ext, ".json") == 0) content_type = "application/json";
            else if (strcmp(ext, ".wav") == 0) content_type = "audio/wav";
            else if (strcmp(ext, ".mp3") == 0) content_type = "audio/mpeg";
            else if (strcmp(ext, ".mp4") == 0) content_type = "video/mp4";
            else content_type = "application/octet-stream";
        } else {
            content_type = "application/octet-stream";
        }

        content = serve_file(filepath, content_type, &content_length);
        response = create_response("200 OK", content_type, content, content_length);
    }
    else if (strncmp(path, "/posts/", 7) == 0) {
        snprintf(filepath, MAX_PATH_LENGTH, "posts/%s", path + 7);
        content = serve_file(filepath, "text/plain", &content_length);
        response = create_response("200 OK", "text/plain", content, content_length);
    }
    else {
        content = "404 Not Found";
        content_length = strlen(content);
        response = create_response("404 Not Found", "text/plain", content, content_length);
    }

    if (response) {
        send(client_socket, response, strlen(response), 0);
        free(response);
    }

    if (content && strcmp(content, "404 Not Found") != 0) {
        free(content);
    }
}