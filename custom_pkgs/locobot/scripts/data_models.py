from dataclasses import dataclass
from typing import List


@dataclass
class GsamRequest:
    """Data sent from client to server."""

    image: list  # RGB image, np.ndarray
    prompt: str


@dataclass
class GsamResponse:
    """Data sent from server to client."""

    masks: List[list] # List of np.ndarray
    labels: List[str] 
    confidences: List[float] 


@dataclass
class AnygraspRequest:
    """
    Request for the AnyGrasp service.
    points: The RGB point cloud as a list of lists (n, 6),
            where each row is (x, y, z, r, g, b).
    """
    points: list


@dataclass
class AnygraspResponse:
    """
    Response from the AnyGrasp service.
    grasps: A list of 4x4 lists representing the grasp poses.
    """
    grasps: List[list]

