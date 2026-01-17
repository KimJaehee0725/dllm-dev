FROM dllm-dev:py311-cuda128-torch270

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

ARG UID=1000
ARG GID=1000
ARG USERNAME=appuser

# Base + dev deps + code-reading tools
RUN apt-get update && apt-get install -y \
    ca-certificates git curl wget openssh-client \
    locales tzdata \
    unzip vim tmux htop gpustat less jq tree \
    build-essential pkg-config \
    libssl-dev libffi-dev \
    ripgrep fd-find bat \
    zsh \
    ncurses-term \
 && locale-gen en_US.UTF-8 \
 && apt-get purge -y 'libnvidia-*' 'nvidia-*' 'cuda-drivers*' || true \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# Locale settings
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Convenience aliases for Ubuntu binaries
RUN ln -sf $(command -v fdfind) /usr/local/bin/fd || true \
 && ln -sf $(command -v batcat) /usr/local/bin/bat || true

# Install latest codex release
RUN wget -qO /tmp/latest.json "https://api.github.com/repos/openai/codex/releases/latest" \
&& export TAG=$(jq -r .tag_name /tmp/latest.json) \
&& wget -q "https://github.com/openai/codex/releases/download/${TAG}/codex-x86_64-unknown-linux-musl.tar.gz" \
&& tar -xzf codex-x86_64-unknown-linux-musl.tar.gz \
&& mv codex-x86_64-unknown-linux-musl /usr/local/bin/codex \
&& chmod +x /usr/local/bin/codex \
&& rm codex-x86_64-unknown-linux-musl.tar.gz /tmp/latest.json

# Create user
RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -m -u ${UID} -g ${GID} -s /bin/zsh ${USERNAME} \
 && chown -R ${UID}:${GID} /workspace

USER ${USERNAME}
ENV HOME=/home/${USERNAME}
ENV PATH="${HOME}/.local/bin:${PATH}"

# Install uv (user install)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python via uv + create symlinks
RUN uv python install 3.12 \
 && mkdir -p ${HOME}/.local/bin \
 && ln -sf $(uv python find 3.12) ${HOME}/.local/bin/python \
 && ln -sf $(uv python find 3.12) ${HOME}/.local/bin/python3

# Git default settings
ARG GIT_NAME="Codex User"
ARG GIT_EMAIL="codex@example.com"
RUN git config --global init.defaultBranch main \
 && git config --global core.editor vim \
 && git config --global pull.rebase false \
 && git config --global user.name "${GIT_NAME}" \
 && git config --global user.email "${GIT_EMAIL}"

# oh-my-zsh + powerlevel10k + plugins
ENV RUNZSH=no CHSH=no KEEP_ZSHRC=yes

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
 && git clone --depth=1 https://github.com/romkatv/powerlevel10k.git \
      ${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}/themes/powerlevel10k \
 && git clone --depth=1 https://github.com/zsh-users/zsh-autosuggestions \
      ${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
 && git clone --depth=1 https://github.com/zsh-users/zsh-syntax-highlighting \
      ${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# tmux config
RUN cat > ${HOME}/.tmux.conf <<'EOF'
set -g default-terminal "tmux-256color"
set -as terminal-features ",xterm-256color:RGB"

set -g mouse on
set -sg escape-time 10
set -g focus-events on
set -g history-limit 200000

setw -g mode-keys vi
bind-key -T copy-mode-vi v send -X begin-selection
bind-key -T copy-mode-vi y send -X copy-selection-and-cancel

unbind '"'
unbind %
bind | split-window -h
bind - split-window -v
bind -n C-h select-pane -L
bind -n C-j select-pane -D
bind -n C-k select-pane -U
bind -n C-l select-pane -R

set -g status on
set -g status-style "bg=default,fg=default"
set -g status-left  " #S:#I.#P "
set -g status-right " #{?client_prefix,#[reverse] PREFIX #[default],} #{pane_current_path} | %Y-%m-%d %H:%M "
set -g window-status-format         " #I:#W "
set -g window-status-current-format " #[reverse]#I:#W#[default] "
set -g status-interval 2
set -g status-left-length  40
set -g status-right-length 120
EOF

# .zshrc - p10k wizard will run on first zsh launch
RUN cat > ${HOME}/.zshrc <<'EOF'
# Enable Powerlevel10k instant prompt (should stay at top)
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="powerlevel10k/powerlevel10k"

plugins=(
  git
  zsh-autosuggestions
  zsh-syntax-highlighting
)

source "$ZSH/oh-my-zsh.sh"

# uv env
[ -f "$HOME/.local/share/uv/env" ] && source "$HOME/.local/share/uv/env"

# code-reading QoL
export PAGER=less
export LESS='-R -F -X'
alias rg='rg --hidden --glob "!.git/*"'
alias ll='ls -al'

# Load p10k config if exists (created after running 'p10k configure')
[[ -f "$HOME/.p10k.zsh" ]] && source "$HOME/.p10k.zsh"
EOF

CMD ["zsh", "-l"]
