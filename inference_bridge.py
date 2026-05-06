#!/usr/bin/env python3
"""
LeRobot Inference Bridge

Communication bridge between Dobot system (Python 3.8) and 
LeRobot inference service (Python 3.10+) via ZMQ.
"""

import zmq
import json
import time
import numpy as np
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeRobotInferenceBridge:
    """Bridge for communicating with LeRobot inference service"""
    
    def __init__(self, service_host: str = "localhost", service_port: int = 5555, timeout: int = 5000):
        self.service_host = service_host
        self.service_port = service_port
        self.timeout = timeout  # milliseconds
        self.context = None
        self.socket = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to the LeRobot inference service"""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)  # Request socket
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
            self.socket.connect(f"tcp://{self.service_host}:{self.service_port}")
            
            # Test connection with ping
            if self.ping():
                self.connected = True
                logger.info(f"Connected to LeRobot service at {self.service_host}:{self.service_port}")
                return True
            else:
                logger.error("Failed to ping LeRobot service")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to LeRobot service: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the service"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.connected = False
        logger.info("Disconnected from LeRobot service")
    
    def ping(self) -> bool:
        """Health check for the service"""
        try:
            request = {'command': 'ping'}
            self.socket.send_json(request)
            response = self.socket.recv_json()
            return response.get('status') == 'ok'
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            return False
    
    def predict(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get action prediction from LeRobot service"""
        if not self.connected:
            logger.error("Not connected to LeRobot service")
            return None
            
        try:
            # Prepare request
            request = {
                'command': 'predict',
                'observation': self._serialize_observation(observation)
            }
            
            # Send request
            self.socket.send_json(request)
            
            # Receive response
            response = self.socket.recv_json()
            
            if response.get('status') == 'success':
                action = np.array(response['action'], dtype=np.float32)
                return action
            else:
                logger.error(f"Prediction failed: {response.get('message', 'Unknown error')}")
                return None
                
        except zmq.Again:
            logger.error("Request timeout - LeRobot service not responding")
            return None
        except Exception as e:
            logger.error(f"Prediction request failed: {e}")
            return None
    
    def _serialize_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize observation for JSON transmission"""
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
                # Handle nested dictionaries (e.g., images)
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
    
    def shutdown_service(self) -> bool:
        """Request service shutdown"""
        if not self.connected:
            return False
            
        try:
            request = {'command': 'shutdown'}
            self.socket.send_json(request)
            response = self.socket.recv_json()
            return response.get('status') == 'shutting_down'
        except Exception as e:
            logger.warning(f"Shutdown request failed: {e}")
            return False


class LeRobotModelWrapper:
    """Wrapper that mimics the original Imitate_Model interface but uses the service"""
    
    def __init__(self, service_host: str = "localhost", service_port: int = 5555):
        self.bridge = LeRobotInferenceBridge(service_host, service_port)
        self.model_loaded = False
        
    def loadModel(self) -> bool:
        """Initialize connection to LeRobot service"""
        if self.bridge.connect():
            self.model_loaded = True
            logger.info("LeRobot service bridge initialized")
            return True
        else:
            logger.error("Failed to connect to LeRobot service")
            return False
    
    def predict(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get prediction using the service bridge"""
        if not self.model_loaded:
            logger.error("Model not loaded (service not connected)")
            return None
            
        return self.bridge.predict(observation)
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'bridge'):
            self.bridge.disconnect()


# Utility functions for integration
def create_lerobot_model(service_host: str = "localhost", service_port: int = 5555):
    """Factory function to create LeRobot model wrapper"""
    model = LeRobotModelWrapper(service_host, service_port)
    return model


def is_service_available(service_host: str = "localhost", service_port: int = 5555, timeout: int = 2000) -> bool:
    """Check if LeRobot service is available"""
    bridge = LeRobotInferenceBridge(service_host, service_port, timeout)
    available = bridge.connect()
    bridge.disconnect()
    return available


if __name__ == "__main__":
    # Test the bridge
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing LeRobot inference bridge...")
        
        # Test connection
        bridge = LeRobotInferenceBridge()
        if bridge.connect():
            print("✅ Connected to service")
            
            # Test prediction with dummy observation
            dummy_obs = {
                'qpos': np.random.randn(14).astype(np.float32),
                'images': {
                    'top': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                }
            }
            
            action = bridge.predict(dummy_obs)
            if action is not None:
                print(f"✅ Got prediction: {action.shape}")
            else:
                print("❌ Prediction failed")
                
            bridge.disconnect()
        else:
            print("❌ Failed to connect to service")
            print("Make sure the LeRobot service is running:")
            print("python lerobot_inference_service.py --model-path /path/to/model")
    else:
        print("Usage: python inference_bridge.py test")
