import numpy as np

from halotools.utils.vector_utilities import project_onto_plane, elementwise_dot, normalized_vectors
from halotools.utils.mcrotations import random_unit_vectors_3d, random_perpendicular_directions

import scipy

def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        # raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
        #                  ' be negative!')
        print('coeffs do not represent an ellipse: b^2 - 4ac must be negative!')
        den = -den

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    try:
        phi = phi % np.pi
    except:
        print("Complex phi. Using abs")
        phi = abs(phi) % np.pi

    return x0, y0, ap, bp, e, phi

# First, make some functions to easily do each of these things

def project_points(centers, pts):
    """
    Assuming pts is of shape Nx3 and centers is of shape Nx3, project the points onto the plane
    perpendicular to the line of sight to the center
    """
    mags = np.linalg.norm(centers, axis=1)
    vertical_3D = normalized_vectors(np.array( [ np.zeros(len(mags)), np.zeros(len(mags)), mags ] ).T - centers)
    horizontal_3D = normalized_vectors(np.cross(centers, vertical_3D))

    vertical_2D = normalized_vectors( project_onto_plane(vertical_3D, centers) )
    horizontal_2D = normalized_vectors( project_onto_plane(horizontal_3D, centers) )

    pts_projected = [ project_onto_plane(pts[i], centers[i]) for i in range(len(centers)) ]

    pts_dot_x = np.array( [ elementwise_dot(pts_projected[i], horizontal_2D[i]) for i in range(len(centers)) ] )
    pts_dot_y = np.array( [ elementwise_dot(pts_projected[i], vertical_2D[i]) for i in range(len(centers)) ] )

    return np.array([pts_dot_x.T, pts_dot_y.T]).T

def ellipsoid_vectorized(center, axis_a, axis_b, axis_c, theta, phi):
    x_piece = center[:,0][:,np.newaxis] + \
                ( np.multiply.outer( axis_a[:,0], np.multiply.outer( np.cos(theta), np.sin(phi) ).ravel() ) ) + \
                ( np.multiply.outer( axis_b[:,0], np.multiply.outer( np.sin(theta), np.sin(phi) ).ravel() ) ) + \
                ( np.multiply.outer( axis_c[:,0], np.multiply.outer( np.ones(len(theta)), np.cos(phi) ).ravel() ) )
    
    y_piece = center[:,1][:,np.newaxis] + \
                ( np.multiply.outer( axis_a[:,1], np.multiply.outer( np.cos(theta), np.sin(phi) ).ravel() ) ) + \
                ( np.multiply.outer( axis_b[:,1], np.multiply.outer( np.sin(theta), np.sin(phi) ).ravel() ) ) + \
                ( np.multiply.outer( axis_c[:,1], np.multiply.outer( np.ones(len(theta)), np.cos(phi) ).ravel() ) )
    
    z_piece = center[:,2][:,np.newaxis] + \
                ( np.multiply.outer( axis_a[:,2], np.multiply.outer( np.cos(theta), np.sin(phi) ).ravel() ) ) + \
                ( np.multiply.outer( axis_b[:,2], np.multiply.outer( np.sin(theta), np.sin(phi) ).ravel() ) ) + \
                ( np.multiply.outer( axis_c[:,2], np.multiply.outer( np.ones(len(theta)), np.cos(phi) ).ravel() ) )
    
    return np.array([x_piece.T, y_piece.T, z_piece.T]).T

def ellipse_vectorized(center, axis_a, axis_b, theta):
    x_piece = center[:,0][:,np.newaxis] + ( np.multiply.outer(axis_a[:,0], np.cos(theta) ) ) + \
                                            ( np.multiply.outer(axis_b[:,0], np.sin(theta) ) )
    y_piece = center[:,1][:,np.newaxis] + ( np.multiply.outer(axis_a[:,1], np.cos(theta) ) ) + \
                                            ( np.multiply.outer(axis_b[:,1], np.sin(theta) ) )
    z_piece = center[:,2][:,np.newaxis] + ( np.multiply.outer(axis_a[:,2], np.cos(theta) ) ) + \
                                            ( np.multiply.outer(axis_b[:,2], np.sin(theta) ) )
    
    return np.array([x_piece.T, y_piece.T, z_piece.T]).T

def generate_ellipse_properties(N=1, unpack=False, Lbox=None, center_Lbox=False):
    if Lbox is None:
        Lbox = np.array([250., 250., 250.])
    a = np.ones(N)
    ratios = np.random.random(size=(N,2))
    ratios.sort(axis=1)
    ratios = ratios[:,::-1]
    b = ratios[:,0]
    c = ratios[:,1]
    unitA = random_unit_vectors_3d(N)
    unitB = random_perpendicular_directions(unitA)
    unitC = normalized_vectors( np.cross(unitA, unitB) )
    axisA = a[:,np.newaxis] * unitA
    axisB = b[:,np.newaxis] * unitB
    axisC = c[:,np.newaxis] * unitC
    if center_Lbox:
        center = np.random.uniform(-Lbox/2, Lbox/2, size=(N,3))
    else:
        center = np.random.uniform(0, Lbox, size=(N,3))

    if N == 1 and unpack:
        return center[0], axisA[0], axisB[0], axisC[0]
    return center, axisA, axisB, axisC

def project_and_select_axes(center, axisA, axisB, axisC):
    mags = np.linalg.norm(center, axis=1)
    vertical_3D = normalized_vectors(np.array( [ np.zeros(len(mags)), np.zeros(len(mags)), mags ] ).T - center)
    horizontal_3D = normalized_vectors(np.cross(center, vertical_3D))           # Define the horizontal axis

    # Reference points
    vertical_2D = project_onto_plane(vertical_3D, center)
    horizontal_2D = project_onto_plane(horizontal_3D, center)

    # Triaxes
    avec_2D = project_onto_plane(axisA, center)
    bvec_2D = project_onto_plane(axisB, center)
    cvec_2D = project_onto_plane(axisC, center)

    # Dot with vertical and horizontal axes
    avec_dot_x = elementwise_dot(avec_2D, horizontal_2D)
    bvec_dot_x = elementwise_dot(bvec_2D, horizontal_2D)
    cvec_dot_x = elementwise_dot(cvec_2D, horizontal_2D)
    avec_dot_y = elementwise_dot(avec_2D, vertical_2D)
    bvec_dot_y = elementwise_dot(bvec_2D, vertical_2D)
    cvec_dot_y = elementwise_dot(cvec_2D, vertical_2D)

    # Pack into 2D vectors
    avec_2D = np.array([avec_dot_x, avec_dot_y]).T
    bvec_2D = np.array([bvec_dot_x, bvec_dot_y]).T
    cvec_2D = np.array([cvec_dot_x, cvec_dot_y]).T

    # Get magnitudes
    avec_mag = np.linalg.norm(avec_2D, axis=1)
    bvec_mag = np.linalg.norm(bvec_2D, axis=1)
    cvec_mag = np.linalg.norm(cvec_2D, axis=1)

    # Get masks for those with the largest magnitude
    bvec_largest = (bvec_mag > avec_mag) & (bvec_mag > cvec_mag)
    cvec_largest = (cvec_mag > avec_mag) & (cvec_mag > bvec_mag)

    # Get largest axis of each
    projected_major = np.array(avec_2D)
    projected_major[bvec_largest] = bvec_2D[bvec_largest]
    projected_major[cvec_largest] = cvec_2D[cvec_largest]

    return projected_major, avec_2D, bvec_2D, cvec_2D

def generate_tri_ellipses(centers, axis_a, axis_b, axis_c, N=10):
    theta = np.linspace(0, 2*np.pi, N)

    AB_ellipse = ellipse_vectorized(centers, axis_a, axis_b, theta)
    BC_ellipse = ellipse_vectorized(centers, axis_b, axis_c, theta)
    AC_ellipse = ellipse_vectorized(centers, axis_a, axis_c, theta)

    return AB_ellipse, BC_ellipse, AC_ellipse

def get_hull_points(pts, multi=True):
    if not multi:
        hull = scipy.spatial.ConvexHull(pts)
        return pts[hull.vertices]
    hulls = [ scipy.spatial.ConvexHull(pts[i]) for i in range(len(pts)) ]
    return [ np.array(pts[i][hulls[i].vertices]) for i in range(len(pts)) ]

def get_ellipse_coefficients(pts, multi=True):
    """
    Returns coefficients for the ellipse that best fits the points
    Coefficients returned are: x0, y0, ap, bp, e, phi
    x0, y0: Center of the ellipse
    ap: Semi-major axis size
    bp: Semi-minor axis size
    e: Eccentricity
    phi: Rotation of the semi-major axis from the x-axis
    """
    if not multi:
        coeffs = fit_ellipse(pts[:,0], pts[:,1])
        coeffs = cart_to_pol(coeffs)
    else:
        coeffs = [ fit_ellipse(pts[i][:,0], pts[i][:,1]) for i in range(len(pts)) ]
        coeffs = np.array([ cart_to_pol(coeff) for coeff in coeffs ])
    return coeffs

def dumbish_a_finder(pts, multi=True):
    """
    Because projection will align the center with 0, simply taking the largest
    magnitude of the points will give the farthest point and therefore major axis.

    It seems there are some slight differences between this and finding the farthest
    two points (speciically that the farthest point away from the center may not be
    part of the pair of farthest two points, but it looks sufficiently close)
    """
    if not multi:
        mags = np.linalg.norm(pts, axis=1)
        return pts[ mags.argmax() ]
    mags = [ np.linalg.norm(pts[i], axis=1) for i in range(len(pts)) ]
    return np.array( [ pts[i][ mags[i].argmax() ] for i in range(len(pts)) ] )