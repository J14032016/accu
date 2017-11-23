# Socket

Socket 是一种操作系统提供的进程间通信机制. 在操作系统中, 通常会为应用程序提供一组应用程序接口, 称为套接字接口(Socket API). 注意的是, Socket API 本身不负责通信, 它仅提供基础函数供应用层调用, 底层通信一般由 [TCP Socket](#tcp-socket) 或 [Unix Socket](#unix-domain-socket) 实现.

# TCP Socket

以下是一个简单的 Socket 服务与其配套客户端实现.

```go
// server.go
package main

import (
    "bufio"
    "log"
    "net"
)

func main() {
    ln, err := net.Listen("tcp", ":3000")
    if err != nil {
        log.Fatalln(err)
    }
    defer ln.Close()
    for {
        conn, err := ln.Accept()
        if err != nil {
            log.Println(err)
            continue
        }
        go func(conn net.Conn) {
            defer conn.Close()
            scanner := bufio.NewScanner(conn)
            for scanner.Scan() {
                log.Println(scanner.Text())
            }
        }(conn)
    }
}
```

```go
// client.go
package main

import (
    "log"
    "net"
)

func main() {
    conn, err := net.Dial("tcp", "127.0.0.1:3000")
    if err != nil {
        panic(err.Error())
    }
    _, err = conn.Write([]byte("accu.cc"))
    if err != nil {
        log.Fatalln(err)
    }
    conn.Close()
}
```

**细节与注意事项**

1. net.Listen 函数 laddr 参数可填写 ":3000", "127.0.0.1:3000", ":0", 其中 ":0" 表示选择任意一个可用端口
2. 使用 `conn.SetDeadLine`, `conn.SetReadDeadLine`, `conn.SetWriteDeadLine` 设置链接超时
3. conn.Read: 如果 client 主动关闭了socket, 且 socket 中有 server 未读取的数据, 则 server 可取出剩余数据, 随后 Read 返回 io.EOF error
4. conn.Read: 如果 client 主动关闭了socket, 且 socket 中无 server 未读取的数据, 则 server Read 返回 io.EOF error
5. conn.Write: 成功写, Write 调用返回的 n 与预期要写入的数据长度相等, 且error == nil
6. conn.Write: 写阻塞, 一端调用 Write 后, 实际上数据是写入到 OS 的协议栈的数据缓冲的. 当另一方未及时 Read 时, 就会发生写阻塞
7. conn.Write: 写入部分数据后, 连接断开, 另一方 Read 的时候, 会取得 n != 0, 且 err != nil, 且 err != io.EOF
8. conn.Write: 写入超时, 当一方设置了 SetWriteDeadline 时, 可能会出现写入超时的情况. 此时 n != 0 且 err 描述为 "i/o timeout"
9. conn.Close: 在己方已关闭的 socket 上执行 read 和 write 操作, 会得到 "use of closed network connection"
10. conn.Close: 在对方已关闭的 socket 上执行 read 操作会得到 io.EOF, 但 write 操作会成功, 因此当发现对方socket关闭后, 己方应该正确合理处理自己的socket

# UNIX Domain Socket

Unix Socket 是 POSIX 操作系统里的一种组件. 它通过文件系统来实现 Socket 通信. 常见的 Unix Socket 文件有 mysql.sock, supervisor.sock 等, 它们均位于 `/var/run/` 目录下.

Go 中使用 Unix Socket 与 TCP Socket 的方法完全相同, 唯一区别是在于服务端使用

```go
net.Listen("unix", "/var/run/accu.sock")
```

启动监听, 客户端使用

```go
net.Dial("unix", "/var/run/accu.sock")
```

与服务端建立链接.
