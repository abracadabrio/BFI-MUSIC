#!/bin/bash

# 进程ID
PID=1659

# 检查进程是否存在
while kill -0 $PID 2> /dev/null; do
    # 进程仍在运行，等待1秒后再检查
    sleep 1
done

# 进程已结束，关闭服务器
shutdown -h now