import os
import sys
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import grpc
from concurrent import futures
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from protos import device_service_pb2 as pb2
from protos import device_service_pb2_grpc as pb2_grpc
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Device %(device_id)s] %(message)s',
    datefmt='%H:%M:%S'
)

class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize weights using PyTorch
        self.W = torch.randn(input_size, output_size, device=self.device) * np.sqrt(2.0 / input_size)
        self.b = torch.zeros(1, output_size, device=self.device)
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)
        
        self.activation = activation
        
    def forward(self, A_prev):
        """Forward pass for a single layer"""
        self.A_prev = A_prev
        self.Z = torch.mm(A_prev, self.W) + self.b
        
        if self.activation == 'relu':
            self.A = F.relu(self.Z)
        else:  # softmax
            self.A = F.softmax(self.Z, dim=1)
        return self.A
    
    def backward(self, dA):
        """Backward pass for a single layer"""
        batch_size = self.A_prev.size(0)
        
        if self.activation == 'relu':
            dZ = dA * (self.Z > 0)
        else:  # softmax
            dZ = dA
            
        self.dW = torch.mm(self.A_prev.t(), dZ) / batch_size
        self.db = torch.sum(dZ, dim=0, keepdim=True) / batch_size
        dA_prev = torch.mm(dZ, self.W.t())
        
        return dA_prev
    
    def update(self, learning_rate):
        """Update parameters for a single layer"""
        with torch.no_grad():
            self.W -= learning_rate * self.dW
            self.b -= learning_rate * self.db

    def calculate_flops(self, A_prev):
        """Calculate FLOPs for the forward pass"""
        # Assuming a fully connected layer
        flops = 2 * A_prev.size(0) * self.W.size(0)  # 2 * (input_size * output_size)
        return flops

    def calculate_backward_flops(self):
        """Calculate FLOPs for the backward pass"""
        # Assuming a fully connected layer
        flops = 2 * self.W.size(0) * self.W.size(1)  # 2 * (output_size * input_size)
        return flops

class Device:
    def __init__(self, layer_configs, device_id=0):
        self.device_id = device_id
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize timing statistics
        self.timing_stats = {
            'forward_compute': [],
            'backward_compute': [],
            'data_transfer': [],
            'parameter_updates': []
        }
        
        # Initialize layers
        self.layers = []
        self.logger.info(f"Initializing device with {len(layer_configs)} layers", extra={'device_id': self.device_id})
        for i, config in enumerate(layer_configs):
            self.logger.info(
                f"Layer {i}: {config['input_size']} -> {config['output_size']} ({config['activation']})",
                extra={'device_id': self.device_id}
            )
            layer = Layer(
                input_size=config['input_size'],
                output_size=config['output_size'],
                activation=config['activation']
            )
            self.layers.append(layer)
        
        self.forward_teraflops = []  # List to store forward teraflops
        self.backward_teraflops = []  # List to store backward teraflops
    
    def quantize(self, x, bits=8):
        """Quantize gradients to reduce communication overhead"""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            
        max_val = np.max(np.abs(x))
        if max_val == 0:
            return x
        
        scale = (2 ** (bits - 1) - 1) / max_val
        quantized = np.round(x * scale)
        return quantized / scale
    
    def forward(self, A_prev):
        """Forward pass through all layers in this device"""
        start_transfer = time.time()
        if isinstance(A_prev, np.ndarray):
            if len(A_prev.shape) == 1:
                A_prev = A_prev.reshape(1, -1)
            elif len(A_prev.shape) == 3:
                A_prev = A_prev.reshape(A_prev.shape[0], -1)
            elif len(A_prev.shape) == 4:
                A_prev = A_prev.reshape(A_prev.shape[0], -1)
            
            A_prev = torch.from_numpy(A_prev).float().to(self.device)
        
        transfer_time = time.time() - start_transfer
        self.timing_stats['data_transfer'].append(transfer_time)
        
        """ self.logger.info(
            f"Forward pass starting with input shape {A_prev.shape}",
            extra={'device_id': self.device_id}
        ) """
        
        # Store activations for backward pass
        self.activations = [A_prev]
        A = A_prev
        
        # Forward through each layer
        start_compute = time.time()
        total_flops = 0  # Initialize FLOPs counter
        for i, layer in enumerate(self.layers):
            layer_start = time.time()
            A = layer.forward(A)
            total_flops += layer.calculate_flops(A_prev)  # Add FLOPs for this layer
            self.activations.append(A)
            layer_time = time.time() - layer_start
        
        compute_time = time.time() - start_compute
        self.timing_stats['forward_compute'].append(compute_time)

        # Calculate and log teraflops
        teraflops = total_flops / compute_time / 1e12  # Convert to TFLOPs
        self.forward_teraflops.append(teraflops)  # Store teraflops
        
        """ self.logger.info(
            f"Forward pass complete - Compute: {compute_time:.4f}s, Transfer: {transfer_time:.4f}s",
            extra={'device_id': self.device_id}
        ) """
        
        return A.detach().cpu().numpy()
    
    def backward(self, dA):
        """Backward pass through all layers in this device"""
        start_transfer = time.time()
        if isinstance(dA, np.ndarray):
            dA = torch.from_numpy(dA).float().to(self.device)
        
        transfer_time = time.time() - start_transfer
        self.timing_stats['data_transfer'].append(transfer_time)
        
        """ self.logger.info(
            f"Backward pass starting with gradient shape {dA.shape}",
            extra={'device_id': self.device_id}
        ) """
        
        # Backward through each layer in reverse
        start_compute = time.time()
        total_flops = 0  # Initialize FLOPs counter
        for i in reversed(range(len(self.layers))):
            layer_start = time.time()
            layer = self.layers[i]
            dA = layer.backward(dA)
            total_flops += layer.calculate_backward_flops()  # Add FLOPs for this layer
            layer_time = time.time() - layer_start
        
        compute_time = time.time() - start_compute
        self.timing_stats['backward_compute'].append(compute_time)

        # Calculate and log teraflops
        teraflops = total_flops / compute_time / 1e12  # Convert to TFLOPs
        self.backward_teraflops.append(teraflops)  # Store teraflops
        
        """ self.logger.info(
            f"Backward pass complete - Compute: {compute_time:.4f}s, Transfer: {transfer_time:.4f}s",
            extra={'device_id': self.device_id}
        ) """
        
        return dA.detach().cpu().numpy()
    
    def update(self, learning_rate):
        """Update parameters of all layers"""
        start_time = time.time()
        """ self.logger.info(
            f"Updating parameters with learning rate {learning_rate}",
            extra={'device_id': self.device_id}
        ) """
        
        for i, layer in enumerate(self.layers):
            layer_start = time.time()
            layer.update(learning_rate)
            layer_time = time.time() - layer_start
            """ self.logger.info(
                f"Updated parameters for layer {i} ({layer_time:.4f}s)",
                extra={'device_id': self.device_id}
            ) """
        
        update_time = time.time() - start_time
        self.timing_stats['parameter_updates'].append(update_time)
        
        # Log average times every 10 updates
        if len(self.timing_stats['parameter_updates']) % 10 == 0:
            avg_forward = np.mean(self.timing_stats['forward_compute'][-10:])
            avg_backward = np.mean(self.timing_stats['backward_compute'][-10:])
            avg_transfer = np.mean(self.timing_stats['data_transfer'][-10:])
            avg_update = np.mean(self.timing_stats['parameter_updates'][-10:])
            
            avg_forward_tflops = np.mean(self.forward_teraflops[-10:]) if self.forward_teraflops else 0
            avg_backward_tflops = np.mean(self.backward_teraflops[-10:]) if self.backward_teraflops else 0
            
            self.logger.info(
                f"Performance stats (last 10 operations):\n"
                f"  Forward compute: {avg_forward:.4f}s\n"
                f"  Backward compute: {avg_backward:.4f}s\n"
                f"  Data transfer: {avg_transfer:.4f}s\n"
                f"  Parameter updates: {avg_update:.4f}s\n"
                f"  Average Forward TFLOPs: {avg_forward_tflops:.4f}\n"
                f"  Average Backward TFLOPs: {avg_backward_tflops:.4f}",
                extra={'device_id': self.device_id}
            )

    def get_teraflops(self):
        """Return the teraflops data for this device."""
        return {
            'forward_tflops': np.mean(self.forward_teraflops) if self.forward_teraflops else 0,
            'backward_tflops': np.mean(self.backward_teraflops) if self.backward_teraflops else 0
        }

class DeviceServicer(pb2_grpc.DeviceServiceServicer):
    def __init__(self):
        self.device = None
    
    def Initialize(self, request, context):
        """Initialize device with layer configurations"""
        layer_configs = [
            {
                'input_size': config.input_size,
                'output_size': config.output_size,
                'activation': config.activation
            }
            for config in request.layer_configs
        ]
        
        self.device = Device(layer_configs, device_id=request.device_id)
        print(f"Initialized device with ID: {self.device.device_id}")  # Log device ID
        
        return pb2.InitResponse(
            status='initialized',
            device_id=self.device.device_id
        )
    
    def Forward(self, request, context):
        """Forward pass through device layers"""
        input_array = np.frombuffer(request.input_data, dtype=np.float32).reshape(request.input_shape)
        output = self.device.forward(input_array)
        
        return pb2.ForwardResponse(
            output_data=output.tobytes(),  # Serialize numpy array directly
            output_shape=list(output.shape)
        )
    
    def Backward(self, request, context):
        """Backward pass through device layers"""
        grad_input = np.frombuffer(request.grad_data, dtype=np.float32).reshape(request.grad_shape)
        grad_output = self.device.backward(grad_input)
        
        return pb2.BackwardResponse(
            grad_output_data=grad_output.tobytes(),  # Serialize numpy array directly
            grad_shape=list(grad_output.shape)
        )
    
    def Update(self, request, context):
        """Update device parameters"""
        self.device.update(request.learning_rate)
        return pb2.UpdateResponse(status='updated')
    
    def Ping(self, request, context):
        """Health check"""
        return pb2.PingResponse(status='connection successful')
    
    def GetTeraflops(self, request, context):
        """Get teraflops data for the device."""
        if self.device is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Device not initialized')
            return pb2.TeraflopsResponse()

        teraflops_data = self.device.get_teraflops()
        print(f"Device {self.device.device_id} - Forward TFLOPs: {teraflops_data['forward_tflops']}, Backward TFLOPs: {teraflops_data['backward_tflops']}")
        return pb2.TeraflopsResponse(
            forward_tflops=teraflops_data['forward_tflops'],
            backward_tflops=teraflops_data['backward_tflops']
        )

def serve(port):
    """Start gRPC server"""
    try:
        print(f"Creating gRPC server...")
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        
        print(f"Adding servicer...")
        pb2_grpc.add_DeviceServiceServicer_to_server(
            DeviceServicer(), server
        )
        
        # Try to bind to the port
        address = f'0.0.0.0:{port}'
        print(f"Binding to address: {address}")
        port = server.add_insecure_port(address)
        
        # Start the server
        print("Starting server...")
        server.start()
        print(f"Device server successfully started and listening on {address}")
        
        # Keep the server running
        server.wait_for_termination()
        
    except Exception as e:
        print(f"Failed to start server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python device_server.py <port>", file=sys.stderr)
        sys.exit(1)
    
    try:
        port = int(sys.argv[1])
        print(f"Attempting to start server on port {port}...")
        serve(port)
    except ValueError:
        print(f"Invalid port number: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)