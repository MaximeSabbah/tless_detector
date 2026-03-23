"""
Publish a MESH_RESOURCE marker so RViz renders the T-LESS CAD model
at the pose estimated by FoundationPose.

The script subscribes to both /tracking/output and /pose_estimation/output
and uses whichever is active. It publishes a MarkerArray on /debug/mesh_marker.

Run:
    source /opt/ros/jazzy/setup.bash
    python3 scripts/visualize_mesh.py

In RViz:
    Add → MarkerArray → topic /debug/mesh_marker
"""
import sys
sys.path.insert(0, '/opt/ros/jazzy/lib/python3.12/site-packages')

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

MESH_PATH = '/workspaces/isaac_ros_ws/isaac_ros_assets/tless_meshes/obj_000023.stl'
MESH_URI  = f'file://{MESH_PATH}'


class MeshVisualizer(Node):
    def __init__(self):
        super().__init__('mesh_visualizer')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pub = self.create_publisher(MarkerArray, '/debug/mesh_marker', 1)

        # Subscribe to both modes; use whichever publishes
        self.create_subscription(
            Detection3DArray, '/tracking/output', self.callback, qos)
        self.create_subscription(
            Detection3DArray, '/pose_estimation/output', self.callback, qos)

        # Publish a DELETE marker on startup to clear any stale markers
        self._clear()
        self.get_logger().info(
            f'Mesh visualizer ready — add MarkerArray /debug/mesh_marker in RViz')

    def _clear(self):
        ma = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        ma.markers.append(m)
        self.pub.publish(ma)

    def callback(self, msg: Detection3DArray):
        if not msg.detections:
            self._clear()
            return

        ma = MarkerArray()
        for i, det in enumerate(msg.detections):
            pose = det.results[0].pose.pose if det.results else None
            if pose is None:
                continue

            m = Marker()
            m.header      = msg.header
            m.ns          = 'foundationpose_mesh'
            m.id          = i
            m.type        = Marker.MESH_RESOURCE
            m.action      = Marker.ADD
            m.mesh_resource = MESH_URI

            m.pose = pose

            # Scale 1:1 — mesh is already in metres
            m.scale.x = 1.0
            m.scale.y = 1.0
            m.scale.z = 1.0

            # Colour: semi-transparent green
            m.color.r = 0.0
            m.color.g = 0.8
            m.color.b = 0.2
            m.color.a = 0.8

            m.lifetime.sec = 1   # auto-clear after 1 s if no new pose

            ma.markers.append(m)

        self.pub.publish(ma)


def main():
    rclpy.init()
    node = MeshVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
