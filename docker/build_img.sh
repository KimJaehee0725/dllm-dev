image_name=dllm-dev:260115
git_name=KimJaehee
git_email=jaehee_kim@snu.ac.kr

docker build --no-cache \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg USERNAME=$(whoami) \
  --build-arg GIT_NAME="${git_name}" \
  --build-arg GIT_EMAIL="${git_email}" \
  -t $image_name .
