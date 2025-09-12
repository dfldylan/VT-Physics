import socket
import struct
import numpy as np
import os

class PBFClient:
    """
    A client to interact with the PBFServer.
    Handles packing/unpacking of binary messages according to the protocol.
    """
    def __init__(self, host='127.0.0.1', port=56789):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        """Connects to the server."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print(f"Successfully connected to {self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            print(f"Connection failed. Is the server running on {self.host}:{self.port}?")
            return False

    def close(self):
        """Closes the connection."""
        if self.sock:
            self.sock.close()
            self.sock = None
            print("Connection closed.")

    def _send_all(self, data):
        if not self.sock:
            raise ConnectionError("Not connected to server.")
        self.sock.sendall(data)

    def _recv_all(self, n):
        if not self.sock:
            raise ConnectionError("Not connected to server.")
        chunks = []
        bytes_recd = 0
        while bytes_recd < n:
            chunk = self.sock.recv(min(n - bytes_recd, 4096))
            if not chunk:
                raise ConnectionAbortedError("Socket connection broken.")
            chunks.append(chunk)
            bytes_recd += len(chunk)
        return b''.join(chunks)

    def reset(self, particle_radius: float):
        """Sends a Reset command (type 1)."""
        print(f"Sending Reset with particle radius: {particle_radius}")
        # <i for int32, <f for float32 (little-endian)
        msg = struct.pack('<if', 1, particle_radius)
        self._send_all(msg)
        ok, = struct.unpack('<i', self._recv_all(4))
        print(f"Reset response: {'OK' if ok else 'Failed'}")
        return ok == 1

    def add_fluid(self, obj_id: int, positions: np.ndarray, velocities: np.ndarray):
        """Sends an Add Fluid command (type 2)."""
        num_particles = len(positions)
        print(f"Sending Add Fluid: id={obj_id}, N={num_particles}")
        
        # Header: type, id, N
        header = struct.pack('<iii', 2, obj_id, num_particles)
        
        # Payload: positions and velocities as flat byte arrays
        pos_bytes = positions.astype(np.float32).tobytes()
        vel_bytes = velocities.astype(np.float32).tobytes()
        
        self._send_all(header + pos_bytes + vel_bytes)
        ok, = struct.unpack('<i', self._recv_all(4))
        print(f"Add Fluid response: {'OK' if ok else 'Failed'}")
        return ok == 1

    def add_solid(self, obj_id: int, positions: np.ndarray, normals: np.ndarray):
        """Sends an Add Solid command (type 3)."""
        num_particles = len(positions)
        print(f"Sending Add Solid: id={obj_id}, N={num_particles}")

        header = struct.pack('<iii', 3, obj_id, num_particles)
        pos_bytes = positions.astype(np.float32).tobytes()
        nrm_bytes = normals.astype(np.float32).tobytes()

        self._send_all(header + pos_bytes + nrm_bytes)
        ok, = struct.unpack('<i', self._recv_all(4))
        print(f"Add Solid response: {'OK' if ok else 'Failed'}")
        return ok == 1

    def next_frame(self, dt: float):
        """Sends a Next Frame command (type 4) and receives particle data."""
        # print(f"Requesting next frame with dt={dt}")
        msg = struct.pack('<if', 4, dt)
        self._send_all(msg)

        # Response header
        ok, obj_count = struct.unpack('<ii', self._recv_all(8))
        if not ok:
            print("Server failed to process next_frame.")
            return None
        
        # print(f"Receiving data for {obj_count} objects...")
        objects = {}
        for _ in range(obj_count):
            obj_id, num_particles = struct.unpack('<ii', self._recv_all(8))
            
            pos_bytes = self._recv_all(num_particles * 3 * 4)
            vel_bytes = self._recv_all(num_particles * 3 * 4)
            
            positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)
            velocities = np.frombuffer(vel_bytes, dtype=np.float32).reshape(-1, 3)
            
            objects[obj_id] = {'pos': positions, 'vel': velocities}
        
        return objects

    def clear(self):
        """Sends a Clear command (type 5)."""
        print("Sending Clear command...")
        msg = struct.pack('<i', 5)
        self._send_all(msg)
        ok, = struct.unpack('<i', self._recv_all(4))
        print(f"Clear response: {'OK' if ok else 'Failed'}")
        return ok == 1

    def shutdown(self):
        """Sends a Shutdown command (type 9)."""
        print("Sending Shutdown command...")
        msg = struct.pack('<i', 9)
        self._send_all(msg)
        print("Shutdown signal sent. Server should terminate.")


def create_fluid_cube(center, size, particle_dist):
    """Creates a grid of points for a fluid cube."""
    x = np.arange(-size/2, size/2, particle_dist)
    y = np.arange(-size/2, size/2, particle_dist)
    z = np.arange(-size/2, size/2, particle_dist)
    x, y, z = np.meshgrid(x, y, z)
    positions = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T + center
    velocities = np.zeros_like(positions)
    return positions, velocities

def create_solid_plane(center, size, particle_dist):
    """Creates a grid of points for a solid plane."""
    x = np.arange(-size/2, size/2, particle_dist)
    z = np.arange(-size/2, size/2, particle_dist)
    x, z = np.meshgrid(x, z)
    y = np.zeros_like(x)
    positions = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T + center
    normals = np.zeros_like(positions)
    normals[:, 1] = 1.0  # Normals point up (Y-axis)
    return positions, normals

def save_ply(filepath, positions):
    """Saves a point cloud to a .ply file."""
    header = f"""ply
format ascii 1.0
element vertex {len(positions)}
property float x
property float y
property float z
end_header
"""
    with open(filepath, 'w') as f:
        f.write(header)
        np.savetxt(f, positions, fmt='%.6f')

def main():
    # --- Configuration ---
    SERVER_PORT = 45001 # 确保这个端口和你的服务器配置一致
    PARTICLE_RADIUS = 0.1
    PARTICLE_DISTANCE = PARTICLE_RADIUS * 2.0
    
    SIMULATION_STEPS = 200
    TIME_STEP = 0.016

    OUTPUT_DIR = "simulation_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Create Client and Connect ---
    client = PBFClient(port=SERVER_PORT)
    if not client.connect():
        return

    try:
        # 1. Reset the scene
        client.reset(particle_radius=PARTICLE_RADIUS)

        # 2. Create and add objects
        fluid_pos, fluid_vel = create_fluid_cube(center=np.array([0, 2, 0]), size=2.0, particle_dist=PARTICLE_DISTANCE)
        client.add_fluid(obj_id=101, positions=fluid_pos, velocities=fluid_vel)

        solid_pos, solid_nrm = create_solid_plane(center=np.array([0, -1, 0]), size=8.0, particle_dist=PARTICLE_DISTANCE)
        client.add_solid(obj_id=201, positions=solid_pos, normals=solid_nrm)

        # 3. Run simulation loop
        print("\n--- Starting simulation loop ---")
        for i in range(SIMULATION_STEPS):
            print(f"Frame {i+1}/{SIMULATION_STEPS}")
            
            # Request next frame
            objects_data = client.next_frame(dt=TIME_STEP)
            if not objects_data:
                print("Simulation failed. Exiting.")
                break

            # Save received data to .ply files
            for obj_id, data in objects_data.items():
                # We only care about visualizing the fluid particles
                if obj_id == 101:
                    ply_path = os.path.join(OUTPUT_DIR, f"fluid_frame_{i:04d}.ply")
                    save_ply(ply_path, data['pos'])
        
        print(f"--- Simulation finished. Output saved to '{OUTPUT_DIR}' directory. ---")

        # 4. Clear the scene
        client.clear()

    except (ConnectionError, ConnectionAbortedError) as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # 5. Shutdown the server and close the client
        client.shutdown()
        client.close()

if __name__ == "__main__":
    main()
