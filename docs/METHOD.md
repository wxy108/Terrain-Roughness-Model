# Contact-geometry terrain filtering

## Inputs and coordinates

The surface is a calibrated height field $h(x,y)$. The saved model and gait provide projected foot coordinates $\mathbf r_i(\phi)$ and a stance set $S_\phi$. Foot coordinates and terrain heights use meters.

At translation $\mathbf x$ and heading $\theta$, a stance foot samples

$$
h_i=h\!\left(\mathbf x+Q_\theta\mathbf r_i(\phi)\right),\qquad
Q_\theta=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}.
$$

The projected positions come from the saved kinematics, without rounding feet to height-map pixels. Swing feet do not contribute to the statistic.

## Pair-angle statistic

For each unordered stance pair,

$$
d_{ij}=\|\mathbf r_i(\phi)-\mathbf r_j(\phi)\|_2,\qquad
\Delta h_{ij}=h_i-h_j,\qquad
\beta_{ij}=\arctan\!\left(\frac{|\Delta h_{ij}|}{d_{ij}}\right).
$$

The distance is horizontal distance. The height difference is measured on the terrain, rather than taken from the nominal foot z coordinate. Rotation preserves $d_{ij}$.

For this saved bank, every phase has $N$ stance feet and $M=N(N-1)/2$ unordered pairs. No zero-distance pair exists. Very short nonzero pairs are retained, including the approximately 0.00699 mm pair recorded in the sampling protocol; no distance cutoff is tuned.

Let $P=50$, $T=36$ and $L$ be the common number of translation points. The final output is

$$
R_g=\frac{180}{\pi}\left[
\frac{1}{PTL}\sum_{p=0}^{P-1}\sum_{q=0}^{T-1}\sum_{\ell=1}^{L}
\frac{1}{M}\sum_{\substack{i<j\\i,j\in S_{\phi_p}}}
\beta_{ij}(\mathbf x_\ell,\phi_p,\theta_q)^2
\right]^{1/2}.
$$

$\phi_p=2\pi p/50$ and $\theta_q=2\pi q/36$. Values stored as `phase_cycles` are $p/50$, not radians. The endpoint at one full cycle is excluded. The implementation accumulates squared-angle sums and applies one square root after all averaging. Unequal final memory chunks retain their actual sample counts.

## Interpolation

A square cell has normalized coordinates $(u,v)\in[0,1]^2$. With vertex heights $h_{00},h_{10},h_{01},h_{11}$ and the diagonal from $(0,0)$ to $(1,1)$, the surface is

$$
h(u,v)=h_{00}[1-\max(u,v)]+h_{11}\min(u,v)
+\begin{cases}h_{10}(u-v),&u\geq v,\\h_{01}(v-u),&u<v.\end{cases}
$$

This piecewise linear triangle convention is the same convention used in the prior heightfield calculations. A boundary vertex is queried through its adjacent final cell with fractional coordinate one. Valid scanning avoids padding or extrapolated terrain.

## Common valid region

For each axis, let $r_{\min},r_{\max}$ be the coordinate extremes over all projected feet, all 50 phases, all 36 headings and all 30 saved wave/segment combinations. Both stance and swing projections enter these bounds.

For a terrain rectangle with coordinate range $[a,b]$, translation centers lie in

$$
x\in[a-r_{\min},\ b-r_{\max}].
$$

This axis-aligned rectangle is exactly the common valid center region for the rectangular terrain and the finite saved kernel bank. All configurations therefore sample the same area. The outer terrain strips remain terrain and supply heights beneath offset feet.

The scan grid includes both endpoints. For length $D$, its point count is $\lceil D/s\rceil+1$, with maximum requested spacing $s=0.002$ m. On the included terrain this gives 220 × 165 = 36,300 centers, with actual spacing approximately 1.997783 × 1.997039 mm.

## Filtering analogy

Each contact geometry selects sparse surface samples. Each foot pair defines a two-point difference response

$$
g_{ij}(\mathbf x)=\frac{h(\mathbf x+\mathbf r_i)-h(\mathbf x+\mathbf r_j)}{d_{ij}}.
$$

Under a cross-correlation convention this is a sparse two-point linear filter with weights $+1/d_{ij}$ and $-1/d_{ij}$. The complete operator applies arctangent, squaring and mean pooling to this bank of responses. Phase and heading generate the kernel bank. The binary contact mask specifies sampling locations; the final statistic requires the pair structure and nonlinear pooling too.

## Height scaling and numerical precision

The nonflat fields use $h_a=\mu+a(h_1-\mu)$, so every interpolated height difference satisfies $\Delta h_a=a\Delta h_1$. The reproduction script therefore shares sampling of the base field across nine amplitudes. It evaluates $\arctan(a|\Delta h_1|/d)$ separately for each amplitude. Final RMS is not a linear function of amplitude.

Coordinates, interpolation, distances and sums use float64. The pair slope is then cast to float32 for arctangent and squaring, matching the validated previous implementation. Compiled operation ordering can introduce sub-microdegree differences. Acceptance tests use a 0.00001° tolerance against independent analytic checks and retained complete-scan results.

## Interpretation

This function returns a gait-conditioned geometric quantity. Neutral-configuration reference roughness is held fixed in the previous terrain/performance plot and remains a separate definition. Its retained CSV column is contextual data, not an output recomputed by this function.

The saved wave-number comparison also changes calibrated amplitude. The saved segment-count comparison keeps segment dimensions and changes total body length. The terrain-height series scales one Libo topography. These conditions support the reported geometric trends within the saved experiment; they do not isolate each variable causally or establish a dynamics-based stability criterion.
