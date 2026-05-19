FROM node:20-slim

WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml* pnpm-workspace.yaml* ./

# Install dependencies
# Note: Since pnpm is not in the path of the host, we install it in the container
RUN npm install -g pnpm && pnpm install

# Copy source code
COPY . .

# Expose Vite port
EXPOSE 5173

# Run dev server
# Use --host to allow external access (from host Windows)
CMD ["pnpm", "dev", "--host"]
