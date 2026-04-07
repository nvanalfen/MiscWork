import numpy as np
from ellipse_projections import ellipsoid_vectorized, project_points, get_hull_points, get_ellipse_coefficients

def project_ellipsoid(centers, vecA, vecB, vecC, theta, phi):
    """
    Project a 3D ellipsoid onto a 2D plane defined by the input vectors.
    """
    # Generate the ellipsoid surface points
    ellipsoids = ellipsoid_vectorized(centers, vecA, vecB, vecC, theta, phi)
    # Project the ellipsoid surface points onto the plane
    ellipses = project_points(centers, ellipsoids)
    # Get the convex hull of the projected points
    hulls = get_hull_points(ellipses)
    coeffs = get_ellipse_coefficients(hulls)
    return np.array(coeffs)

def compute_ellipse2d(a, b, c, pos, major_axes, inter_axes, minor_axes, Nphi=100, Ntheta=100):
    theta = np.linspace(0, np.pi, Ntheta)
    phi = np.linspace(0, 2*np.pi, Nphi)

    coeffs = project_ellipsoid(pos, major_axes*a[:, np.newaxis],
                              inter_axes*b[:, np.newaxis],
                              minor_axes*c[:, np.newaxis],
                              theta, phi)
    
    # Coefficients returned are: x0, y0, ap, bp, e, phi
    # x0, y0: Center of the ellipse
    # ap: Semi-major axis size
    # bp: Semi-minor axis size
    # e: Eccentricity
    # phi: Rotation of the semi-major axis from the x-axis

    x0, y0, ap, bp, e, ellipse_phi = coeffs.T
    # e_alpha = np.array([np.cos(ellipse_phi), np.sin(ellipse_phi)]).T
    # e_beta = np.array([-np.sin(ellipse_phi), np.cos(ellipse_phi)]).T
    # # Flip alpha and beta to match the format in the other equation
    # return {"e_alpha":e_alpha, "e_beta":e_beta, "alpha":ap, "beta":bp, "ellipticity":e}
    return ellipse_phi, e