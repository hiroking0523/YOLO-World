# Xサーバーへのアクセスを現在のユーザーに限定して許可する
xhost +

# Dockerコンテナを起動し、現在のユーザーのDISPLAY環境変数とX11のソケットを渡す
sudo docker run -it --rm \
  --runtime nvidia \
  --shm-size=1g \
  -v $(pwd):/workspace \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --network host \
  yolo-world:latest