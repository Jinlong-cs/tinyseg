FROM ubuntu:22.04

ARG USER_UID=1000
ARG USER_GID=1000
ARG VIDEO_GID=44
ARG USER_NAME=dev
ARG DIR_NAME=project
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo build-essential git git-lfs openssh-client \
    nano vim less tree htop tmux curl wget zip unzip \
    python3 python3-pip python3-venv \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa \
 && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
  PRIMARY_GID="${USER_GID}"; \
  PRIMARY_G="$(getent group "${PRIMARY_GID}" | cut -d: -f1 || true)"; \
  if [ -n "${PRIMARY_G}" ]; then \
    useradd -m -u "${USER_UID}" -g "${PRIMARY_GID}" -G sudo -s /bin/bash "${USER_NAME}"; \
  else \
    groupadd -g "${PRIMARY_GID}" "${USER_NAME}"; \
    useradd -m -u "${USER_UID}" -g "${PRIMARY_GID}" -G sudo -s /bin/bash "${USER_NAME}"; \
  fi; \
  echo "%sudo ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/nopasswd; \
  chmod 0440 /etc/sudoers.d/nopasswd

RUN set -eux; \
  TARGET_GID="${VIDEO_GID}"; \
  EXIST_G="$(getent group "${TARGET_GID}" | cut -d: -f1 || true)"; \
  if [ -n "${EXIST_G}" ]; then \
    usermod -aG "${EXIST_G}" "${USER_NAME}"; \
  else \
    groupadd -g "${TARGET_GID}" video; \
    usermod -aG video "${USER_NAME}"; \
  fi

RUN python3 -m pip install --no-cache-dir --upgrade pip uv

USER ${USER_NAME}
WORKDIR /${DIR_NAME}

RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
