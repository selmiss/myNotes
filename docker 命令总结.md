## docker 命令总结

### 前置

> docker安装命令

```shell
curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
```

```
curl -sSL https://get.daocloud.io/docker | sh
```

### 1. docker基础命令

> 开启关闭

```shell
systemctl start docker //启动docker
systemctl stop docker //关闭docker
systemctl restart docker //重启docker
systemctl enable docker //设置随服务启动自启动
systemctl status docker	//查看docker运行状态
```

> 查看版本信息/帮助

```shell
docker version
docker info
docker --help
```

### 2. docker镜像命令

> 查看镜像列表

```
docker images
```

> 搜索镜像

```shell
docker search <镜像名>
docker search --filter=STARS=9000 mysql #搜索 STARS >9000的 mysql 镜像
```

>  拉取镜像,tag为拉取指定版本，默认最新版本

```shell
docker pull 镜像名 
docker pull 镜像名:tag
```

> **运行镜像**

```shell
docker run 镜像名
docker run 镜像名:Tag
```

> 删除镜像

```shell
#删除一个
docker rmi -f 镜像名/镜像ID

#删除多个 其镜像ID或镜像用用空格隔开即可 
docker rmi -f 镜像名/镜像ID 镜像名/镜像ID 镜像名/镜像ID

#删除全部镜像  -a 意思为显示全部, -q 意思为只显示ID
docker rmi -f $(docker images -aq)

#强制删除
docker image rm 镜像名称/镜像ID
```

> 保存镜像、加载镜像

将我们的镜像 保存为tar 压缩文件 这样方便镜像转移和保存

```
docker save 镜像名/镜像ID -o 镜像保存在哪个位置与名字
docker load -i 镜像保存文件位置
```

### 3.docker容器命令

> 查看正在运行容器列表/所有容器

```
docker ps
docker ps -a
```

> 运行一个容器

```shell
# -it 表示 与容器进行交互式启动 
#-d 表示可后台运行容器 （守护式运行）  
# --name 给要运行的容器 起的名字  
# /bin/bash  交互路径
# -p 绑定端口 宿主机端口:容器端口
# --restart=always 自启动
# 挂载文件 {-v 宿主机文件存储位置:容器内文件位置}
docker run -itd --name 要取的别名 -p 宿主机端口:容器端口 --restart=always 镜像名:Tag /bin/bash 
```

> 停止/启动/重启/杀死 容器

```
docker stop 容器名/容器ID
docker start 容器ID/容器名
docker restart 容器ID/容器名
docker kill 容器ID/容器名
```

> 删除容器

```shell
#删除一个容器
docker rm -f 容器名/容器ID
#删除多个容器 空格隔开要删除的容器名或容器ID
docker rm -f 容器名/容器ID 容器名/容器ID 容器名/容器ID
#删除全部容器
docker rm -f $(docker ps -aq)
```

> 进入容器exec/attach

```shell
docker exec -it 容器名/容器ID /bin/bash

docker attach 容器名/容器ID
```

> 退出容器

```shell
#-----直接退出  未添加 -d(持久化运行容器) 时 执行此参数 容器会被关闭  
exit
# 优雅退出 --- 无论是否添加-d 参数 执行此命令容器都不会被关闭
Ctrl + p + q
```

> 容器文件拷贝

````shell
#从容器内 拷出
docker cp 容器ID/名称: 容器内路径  容器外路径
#从外部 拷贝文件到容器内
docker  cp 容器外路径 容器ID/名称: 容器内路径
````

> 查看容器日志

```shell
docker logs -f --tail=要查看末尾多少行 容器ID
```

> 修改容器启动配置

```
docker  update --restart=always 容器Id 或者 容器名
docker container update --restart=always 容器Id 或者 容器名
```

> 更换容器名

```
docker rename 容器ID/容器名 新容器名
```

### 4.提交镜像

> 构建新的镜像

```shell
docker commit -m="提交信息" -a="作者信息" 容器名/容器ID 提交后的镜像名:Tag
```

### 5.docker运维命令

> 查看docker工作目录

```shell
sudo docker info | grep "Docker Root Dir"
```

> 查看磁盘占用情况

```shell
#总体情况
du -hs /var/lib/docker/  
#具体情况
docker system df
```

> 删除无用的容器和镜像

```shell
#  删除异常停止的容器
docker rm `docker ps -a | grep Exited | awk '{print $1}'` 
 
#  删除名称或标签为none的镜像
docker rmi -f  `docker images | grep '<none>' | awk '{print $3}'`
```

### 6.参考文献

https://blog.csdn.net/leilei1366615/article/details/106267225
