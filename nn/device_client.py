import grpc
import subprocess
import requests
import sys
import time
import signal
import os
from protos import device_service_pb2 as pb2
from protos import device_service_pb2_grpc as pb2_grpc

def get_local_ip():
    """Get the local IP address of this machine"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to an external server (doesn't actually send anything)
        s.connect(('8.8.8.8', 80))
        IP = s.getsockname()[0]
    except Exception:
        # Fallback to localhost if unable to determine IP
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

class DeviceClient:
    def __init__(self, api_url="http://localhost:4000", device_port=None, job_id=None):
        self.api_url = api_url
        self.device_port = device_port or self._find_available_port(start_port=5001)
        self.job_id = job_id
        self.device_process = None
        self.local_ip = get_local_ip()
        
    def _find_available_port(self, start_port):
        """Find an available port starting from start_port"""
        import socket
        
        port = start_port
        while port < start_port + 100:  # Try up to 100 ports
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                # Bind to all interfaces instead of just localhost
                sock.bind(('0.0.0.0', port))
                sock.close()
                return port
            except OSError:
                port += 1
            finally:
                sock.close()
        raise RuntimeError("Could not find an available port")

    def start_device_server(self):
        """Start the device server as a subprocess"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            server_path = os.path.join(current_dir, 'device_server.py')
            
            print(f"Starting device server on port {self.device_port}...")
            
            # Start server without pipes to see output directly
            self.device_process = subprocess.Popen(
                [sys.executable, server_path, str(self.device_port)]
            )
            
            # Wait and check if process is still running
            time.sleep(2)
            if self.device_process.poll() is not None:
                raise RuntimeError("Device server process terminated unexpectedly")
            
            # Wait for server to be ready
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    print(f"Attempting to connect to device server (attempt {attempt + 1}/{max_attempts})...")
                    with grpc.insecure_channel(f'0.0.0.0:{self.device_port}') as channel:
                        channel_ready = grpc.channel_ready_future(channel)
                        channel_ready.result(timeout=2)  # Wait for channel to be ready
                        
                        stub = pb2_grpc.DeviceServiceStub(channel)
                        response = stub.Ping(pb2.PingRequest())
                        if response.status == 'connection successful':
                            print(f"Successfully connected to device server on {self.local_ip}:{self.device_port}")
                            return True
                except Exception as e:
                    if attempt < max_attempts - 1:
                        print(f"Connection attempt failed: {e}")
                        time.sleep(2)
                    else:
                        raise RuntimeError(f"Failed to connect after {max_attempts} attempts")
            
            return False
                
        except Exception as e:
            print(f"Error starting device server: {e}")
            self.cleanup()
            return False

    def register_with_api(self):
        """Register this device with the API server"""
        try:
            response = requests.post(
                f"{self.api_url}/api/devices/register",
                json={
                    'port': self.device_port,
                    'ip': self.local_ip,
                    'job_id': self.job_id
                }
            )
            
            if response.status_code == 201:
                print("Successfully registered with API server")
                return response.json()['device_id']
            else:
                print(f"Failed to register: {response.json()['error']}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to API server: {e}")
            return None

    def cleanup(self):
        """Clean up resources and stop the device server"""
        if self.device_process:
            try:
                # Try to unregister from the API
                requests.delete(f"{self.api_url}/api/devices/{self.device_port}")
            except:
                pass
                
            # Kill the device server process
            self.device_process.terminate()
            try:
                self.device_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.device_process.kill()
            
            self.device_process = None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Start a neural network device server and register with API')
    parser.add_argument('--api-url', default='http://localhost:4000', help='API server URL')
    parser.add_argument('--port', type=int, help='Port for device server (optional)')
    parser.add_argument('--job-id', required=True, help='ID of the job to join')
    args = parser.parse_args()
    
    client = DeviceClient(api_url=args.api_url, device_port=args.port, job_id=args.job_id)
    
    def signal_handler(signum, frame):
        print("\nReceived shutdown signal. Cleaning up...")
        client.cleanup()
        sys.exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if client.start_device_server():
            device_id = client.register_with_api()
            if device_id:
                print(f"Device {device_id} is running. Press Ctrl+C to stop.")
                # Keep the main thread alive
                while True:
                    time.sleep(1)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.cleanup()

if __name__ == "__main__":
    main() 