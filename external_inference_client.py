#!/usr/bin/env python3
"""
External LeRobot Inference Client

Unified client supporting multiple protocols (TCP, HTTP, ZMQ) for communicating 
with an external LeRobot inference server.
Sends observations (joint positions + images) and receives joint position commands.
"""

import json
import time
import numpy as np
from typing import Dict, Any, Optional, Union
import logging
import cv2
import base64
import socket
import struct
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import requests
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False


class TCPInferenceClient:
    """TCP-based client for external LeRobot inference server"""
    
    def __init__(self, server_host: str = "localhost", server_port: int = 9999, timeout: float = 5.0):
        self.server_host = server_host
        self.server_port = server_port
        self.timeout = timeout
        self.socket = None
        self.connected = False
        self.lock = threading.Lock()
        
    def connect(self) -> bool:
        """Connect to the TCP server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.server_host, self.server_port))
            
            # Test connection with ping
            if self._ping():
                self.connected = True
                logger.info(f"Connected to TCP server at {self.server_host}:{self.server_port}")
                return True
            else:
                self.disconnect()
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to TCP server: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None
            return False
    
    def disconnect(self):
        """Disconnect from the server"""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        logger.info("Disconnected from TCP server")
    
    def _ping(self) -> bool:
        """Test connection with ping"""
        try:
            ping_msg = {"command": "ping"}
            response = self._send_message(ping_msg)
            return response and response.get('status') == 'ok'
        except Exception as e:
            logger.warning(f"TCP ping failed: {e}")
            return False
    
    def predict(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Send observation and get joint position prediction"""
        if not self.connected:
            logger.warning("TCP client not connected")
            return None
        
        with self.lock:  # Thread-safe access
            try:
                # Prepare message
                message = {
                    "command": "predict",
                    "observation": self._prepare_observation(observation)
                }
                
                # Send and receive
                response = self._send_message(message)
                
                if response and response.get('status') == 'success':
                    joint_positions = np.array(response['joint_positions'], dtype=np.float32)
                    return joint_positions
                else:
                    logger.error(f"TCP prediction failed: {response.get('message', 'Unknown error') if response else 'No response'}")
                    return None
                    
            except Exception as e:
                logger.error(f"TCP prediction request failed: {e}")
                self.connected = False
                return None
    
    def _send_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send JSON message with length prefix and receive response"""
        try:
            # Serialize message
            json_data = json.dumps(message).encode('utf-8')
            
            # Send length prefix (4 bytes, big-endian) followed by message
            length = len(json_data)
            self.socket.sendall(struct.pack('>I', length))
            self.socket.sendall(json_data)
            
            # Receive length prefix
            length_data = self._recv_exactly(4)
            if not length_data:
                return None
            
            response_length = struct.unpack('>I', length_data)[0]
            
            # Receive response
            response_data = self._recv_exactly(response_length)
            if not response_data:
                return None
            
            # Deserialize response
            response = json.loads(response_data.decode('utf-8'))
            return response
            
        except socket.timeout:
            logger.error("TCP socket timeout")
            return None
        except Exception as e:
            logger.error(f"TCP message send/receive failed: {e}")
            return None
    
    def _recv_exactly(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes from socket"""
        data = b''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def _prepare_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare observation for transmission"""
        prepared = {}
        
        # Handle joint positions
        if 'qpos' in observation:
            qpos = observation['qpos']
            if isinstance(qpos, np.ndarray):
                prepared['joint_positions'] = qpos.tolist()
            else:
                prepared['joint_positions'] = list(qpos)
        
        # Handle images - encode as base64 JPEG for efficiency
        if 'images' in observation:
            prepared['images'] = {}
            for camera_name, image in observation['images'].items():
                if isinstance(image, np.ndarray):
                    encoded_image = self._encode_image(image)
                    prepared['images'][camera_name] = encoded_image
        
        return prepared
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image as base64 JPEG string"""
        try:
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                if image.max() <= 1.0:  # Normalized image
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Convert to base64
            encoded = base64.b64encode(buffer).decode('utf-8')
            return encoded
            
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return ""


class HTTPInferenceClient:
    """HTTP-based client for external LeRobot inference server"""
    
    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 5):
        if not HTTP_AVAILABLE:
            raise ImportError("HTTP client requires 'requests' package. Install with: pip install requests")
            
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.connected = False
        
    def connect(self) -> bool:
        """Test connection to the inference server"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=self.timeout)
            if response.status_code == 200:
                self.connected = True
                logger.info(f"Connected to inference server at {self.server_url}")
                return True
            else:
                logger.error(f"Server responded with status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    def predict(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Send observation and get joint position prediction"""
        if not self.connected:
            logger.warning("Not connected to server, attempting to connect...")
            if not self.connect():
                return None
        
        try:
            # Prepare payload
            payload = self._prepare_payload(observation)
            
            # Send request
            response = self.session.post(
                f"{self.server_url}/predict", 
                json=payload, 
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    joint_positions = np.array(result['joint_positions'], dtype=np.float32)
                    return joint_positions
                else:
                    logger.error(f"Prediction failed: {result.get('message', 'Unknown error')}")
                    return None
            else:
                logger.error(f"Server error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Request timeout - server not responding")
            return None
        except Exception as e:
            logger.error(f"Prediction request failed: {e}")
            return None
    
    def _prepare_payload(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare observation payload for the server"""
        payload = {
            'observation': {}
        }
        
        # Handle joint positions
        if 'qpos' in observation:
            qpos = observation['qpos']
            if isinstance(qpos, np.ndarray):
                payload['observation']['joint_positions'] = qpos.tolist()
            else:
                payload['observation']['joint_positions'] = list(qpos)
        
        # Handle images
        if 'images' in observation:
            payload['observation']['images'] = {}
            for camera_name, image in observation['images'].items():
                if isinstance(image, np.ndarray):
                    # Encode image as base64 JPEG
                    encoded_image = self._encode_image(image)
                    payload['observation']['images'][camera_name] = encoded_image
        
        return payload
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image as base64 JPEG string"""
        try:
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                if image.max() <= 1.0:  # Normalized image
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Convert to base64
            encoded = base64.b64encode(buffer).decode('utf-8')
            return encoded
            
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return ""


class SimpleInferenceWrapper:
    """Simple wrapper that mimics the original model interface"""
    
    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 5):
        self.client = ExternalInferenceClient(server_url, timeout)
        self.model_loaded = False
        
    def loadModel(self) -> bool:
        """Initialize connection to external server"""
        if self.client.connect():
            self.model_loaded = True
            logger.info("External inference client initialized")
            return True
        else:
            logger.error("Failed to connect to external inference server")
            return False
    
    def predict(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get prediction from external server"""
        if not self.model_loaded:
            logger.error("Client not initialized (server not connected)")
            return None
            
        return self.client.predict(observation)


# Alternative: ZMQ-based client (if your server uses ZMQ instead of HTTP)
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False


class ZMQInferenceClient:
    """ZMQ-based client for external inference server"""
    
    def __init__(self, server_host: str = "localhost", server_port: int = 5555, timeout: int = 5000):
        if not ZMQ_AVAILABLE:
            raise ImportError("ZMQ not available. Install with: pip install pyzmq")
            
        self.server_host = server_host
        self.server_port = server_port
        self.timeout = timeout
        self.context = None
        self.socket = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to ZMQ server"""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
            self.socket.connect(f"tcp://{self.server_host}:{self.server_port}")
            
            # Test connection
            test_msg = {"command": "ping"}
            self.socket.send_json(test_msg)
            response = self.socket.recv_json()
            
            if response.get('status') == 'ok':
                self.connected = True
                logger.info(f"Connected to ZMQ server at {self.server_host}:{self.server_port}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"ZMQ connection failed: {e}")
            return False
    
    def predict(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Send observation via ZMQ and get prediction"""
        if not self.connected:
            return None
            
        try:
            # Prepare message
            message = {
                "command": "predict",
                "observation": self._serialize_observation(observation)
            }
            
            # Send and receive
            self.socket.send_json(message)
            response = self.socket.recv_json()
            
            if response.get('status') == 'success':
                joint_positions = np.array(response['joint_positions'], dtype=np.float32)
                return joint_positions
            else:
                logger.error(f"ZMQ prediction failed: {response.get('message')}")
                return None
                
        except zmq.Again:
            logger.error("ZMQ timeout")
            return None
        except Exception as e:
            logger.error(f"ZMQ prediction error: {e}")
            return None
    
    def _serialize_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize observation for ZMQ transmission"""
        serialized = {}
        
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                serialized[key] = {
                    'type': 'ndarray',
                    'data': value.tolist(),
                    'dtype': str(value.dtype),
                    'shape': value.shape
                }
            elif isinstance(value, dict):
                # Handle nested dicts (e.g., images)
                serialized[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        serialized[key][sub_key] = {
                            'type': 'ndarray',
                            'data': sub_value.tolist(),
                            'dtype': str(sub_value.dtype),
                            'shape': sub_value.shape
                        }
                    else:
                        serialized[key][sub_key] = sub_value
            else:
                serialized[key] = value
                
        return serialized
    
    def disconnect(self):
        """Disconnect from ZMQ server"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.connected = False


# Factory functions
def create_tcp_client(server_host: str = "localhost", server_port: int = 9999, timeout: float = 5.0):
    """Create TCP-based inference client (recommended - no dependencies)"""
    class TCPWrapper:
        def __init__(self):
            self.client = TCPInferenceClient(server_host, server_port, timeout)
            self.model_loaded = False
            
        def loadModel(self):
            if self.client.connect():
                self.model_loaded = True
                return True
            return False
            
        def predict(self, observation):
            if not self.model_loaded:
                return None
            return self.client.predict(observation)
            
        def __del__(self):
            if hasattr(self, 'client'):
                self.client.disconnect()
    
    return TCPWrapper()


def create_http_client(server_url: str = "http://localhost:8000", timeout: int = 5):
    """Create HTTP-based inference client"""
    if not HTTP_AVAILABLE:
        raise ImportError("HTTP client requires 'requests' package. Install with: pip install requests")
    
    class HTTPWrapper:
        def __init__(self):
            self.client = HTTPInferenceClient(server_url, timeout)
            self.model_loaded = False
            
        def loadModel(self):
            if self.client.connect():
                self.model_loaded = True
                return True
            return False
            
        def predict(self, observation):
            if not self.model_loaded:
                return None
            return self.client.predict(observation)
    
    return HTTPWrapper()


def create_zmq_client(server_host: str = "localhost", server_port: int = 5555, timeout: int = 5000):
    """Create ZMQ-based inference client"""
    if not ZMQ_AVAILABLE:
        raise ImportError("ZMQ not available. Install with: pip install pyzmq")
    
    class ZMQWrapper:
        def __init__(self):
            self.client = ZMQInferenceClient(server_host, server_port, timeout)
            self.model_loaded = False
            
        def loadModel(self):
            if self.client.connect():
                self.model_loaded = True
                return True
            return False
            
        def predict(self, observation):
            if not self.model_loaded:
                return None
            return self.client.predict(observation)
            
        def __del__(self):
            if hasattr(self, 'client'):
                self.client.disconnect()
    
    return ZMQWrapper()


if __name__ == "__main__":
    # Test the clients
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing external inference clients...")
        
        # Test with dummy observation
        dummy_obs = {
            'qpos': np.random.randn(14).astype(np.float32),
            'images': {
                'top': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            }
        }
        
        # Test TCP client (recommended)
        print("\n1. Testing TCP client...")
        tcp_client = create_tcp_client()
        if tcp_client.loadModel():
            print("✅ TCP client connected")
            
            result = tcp_client.predict(dummy_obs)
            if result is not None:
                print(f"✅ TCP prediction successful: {result.shape}")
            else:
                print("❌ TCP prediction failed")
        else:
            print("❌ TCP client connection failed")
            print("Start TCP server with: python external_inference_client.py server")
        
        # Test HTTP client
        if HTTP_AVAILABLE:
            print("\n2. Testing HTTP client...")
            http_client = create_http_client()
            if http_client.loadModel():
                print("✅ HTTP client connected")
                
                result = http_client.predict(dummy_obs)
                if result is not None:
                    print(f"✅ HTTP prediction successful: {result.shape}")
                else:
                    print("❌ HTTP prediction failed")
            else:
                print("❌ HTTP client connection failed")
        else:
            print("\n2. HTTP not available (install with: pip install requests)")
        
        # Test ZMQ client if available
        if ZMQ_AVAILABLE:
            print("\n3. Testing ZMQ client...")
            try:
                zmq_client = create_zmq_client()
                if zmq_client.loadModel():
                    print("✅ ZMQ client connected")
                    
                    result = zmq_client.predict(dummy_obs)
                    if result is not None:
                        print(f"✅ ZMQ prediction successful: {result.shape}")
                    else:
                        print("❌ ZMQ prediction failed")
                else:
                    print("❌ ZMQ client connection failed")
            except Exception as e:
                print(f"❌ ZMQ test failed: {e}")
        else:
            print("\n3. ZMQ not available (install with: pip install pyzmq)")
            
    else:
        print("Usage: python external_inference_client.py test")
        print("\nExample usage in your code:")
        print("""
# HTTP-based client
client = create_http_client("http://your-server:8000")
client.loadModel()

# ZMQ-based client  
client = create_zmq_client("your-server", 5555)
client.loadModel()

# Use like original model
action = client.predict(observation)
""")
