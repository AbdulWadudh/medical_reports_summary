# Use the official Bun image
FROM oven/bun:latest

# Set the working directory
WORKDIR /app

# Copy package files first for better caching
COPY package.json bun.lockb* ./

# Install dependencies
RUN bun install --frozen-lockfile

# Copy the rest of the application code
COPY . .

# Create the temporary directory for file processing
RUN mkdir -p /tmp/medical-claims && chmod 777 /tmp/medical-claims

# Expose the requested port
EXPOSE 3434

# Set production environment
ENV NODE_ENV=production

# Start the application
CMD ["bun", "run", "start"]
