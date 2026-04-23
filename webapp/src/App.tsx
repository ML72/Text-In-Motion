import React from 'react';
import { Box, Typography, Button, Container, Divider } from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';
import { MathJaxContext, MathJax } from 'better-react-mathjax';

const App: React.FC = () => {
  return (
    <MathJaxContext>
    <Box sx={{ minHeight: '100vh', background: 'radial-gradient(circle at 50% 0%, #ffffff 0%, #f0f4f8 100%)', color: '#0f172a', py: 8 }}>
      <Container maxWidth='md'>
        {/* Header Section */}
        <Box sx={{ textAlign: 'center', mb: 6 }}>
          <Typography variant='h3' component='h1' gutterBottom sx={{ fontWeight: 600, letterSpacing: '-0.02em', background: '-webkit-linear-gradient(45deg, #1976d2, #9c27b0)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Text in Motion: Visualizing Prose as Stylistic Dance Sequences
          </Typography>
          <Typography variant='subtitle1' sx={{ color: 'text.secondary', mt: 2, mb: 1 }}>
            Michael Li and Alison Ding, Carnegie Mellon University
          </Typography>
          <Box sx={{ mt: 4 }}>
            <Button variant='contained' startIcon={<GitHubIcon />} href='https://github.com/ML72/Text-In-Motion' target='_blank' disableElevation sx={{ backgroundColor: '#1976d2', color: 'white', '&:hover': { backgroundColor: '#1565c0' } }}>
              View Code
            </Button>
          </Box>
        </Box>

        <Divider sx={{ borderColor: 'rgba(0,0,0,0.08)', mb: 6 }} />

        {/* Abstract Section */}
        <Box sx={{ mb: 6 }}>
          <Typography variant='h4' component='h2' gutterBottom sx={{ fontWeight: 600 }}>
            Abstract
          </Typography>
          <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8 }}>
            Recent advances in generative artificial intelligence (GenAI) systems have allowed for unprecedented fidelity in synthesizing complex human motion.
            However, current text-to-dance models rely strictly on linguistic semantics to generate movements. For example, if an input prompt or song lyric reads "raise hand", the model outputs that literal action.
            In this work, we depart from the traditional paradigm of linguistic grounding to investigate the capacity for inherent machine creativity.
            To this end, we develop a self-consistent and novel approach to motion synthesis that generates dance by leveraging the abstract structural properties of language rather than literal interpretation.
            Our method first quantizes dance kinematics into a motion codebook of stylistic regions via Principal Component Analysis (PCA) and K-Means Clustering.
            We can then treat the byte-level representation of text as a unique ``DNA fingerprint'' which directly maps to a specific sequence of codebook regions, which we can treat as a deterministic yet novel choreographic blueprint for synthesizing dance sequences.
            To ensure fluid execution, we then evaluate physical plausibility constraints using a graph-based approach, before finally synthesizing a motion sequence adhering to the specified blueprint.
            This framework fundamentally repositions AI from being a literal interpreter of human intent to being a procedural instrument for the algorithmic translation of abstract data into movement.
          </Typography>
        </Box>

        <Divider sx={{ borderColor: 'rgba(0,0,0,0.08)', mb: 6 }} />

        {/* Motivation Section */}
        <Box sx={{ mb: 6 }}>
          <Typography variant='h4' component='h2' gutterBottom sx={{ fontWeight: 600 }}>
            Motivation
          </Typography>
          <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8 }}>
            This project asks a different question: rather than being confined by semantic meaning, can we utilize text as a non-linguistic, structural "DNA fingerprint" to create a novel choreographic score?
            By liberating text from its semantics and treating it as a literal code for a procedural art tool, we challenge the prevailing view in multimodal AI of language as a universal grounding mechanism. Semantically grounded AI often gives relatively predictable outputs constrained by the biases of training data. 
            <br /><br />
            By treating text as an objective raw data stream rather than linguistic intent, we provide artists with a mathematically interpretable, deterministic alternative to black-box neural networks. We seek to reposition AI not as an imitator of human history, but as an autonomous co-creator capable of evolving new, systematic artistic identities by literally transcoding arbitrary data into physical expression.
          </Typography>
        </Box>

        <Divider sx={{ borderColor: 'rgba(0,0,0,0.08)', mb: 6 }} />

        {/* Demos Section */}
        <Box sx={{ mb: 6 }}>
          <Typography variant='h4' component='h2' gutterBottom sx={{ fontWeight: 600 }}>
            Demos
          </Typography>

          {/* Autonomous Exploration */}
          <Box sx={{ mb: 6 }}>
            <Typography variant='h5' component='h3' gutterBottom sx={{ color: 'primary.main', fontWeight: 500 }}>
              Autonomous Exploration
            </Typography>
            <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8, mb: 3 }}>
              Autonomous exploration represents a standard motion matching algorithm with no textual inputs. By sampling new sequences on the fly, it serves as a baseline that achieves high choreographic novelty but lacks the structural guidance of our text-driven approach. Driven almost entirely by random sampling, this dynamic blending generates movement phrases consistently unseen in the original data, unfolding as an organically smooth, abstract art form.
            </Typography>
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 2 }}>
              <video src="videos/auto_male.mp4" autoPlay loop muted playsInline controls style={{ width: '100%', borderRadius: '8px' }} />
              <video src="videos/auto_female.mp4" autoPlay loop muted playsInline controls style={{ width: '100%', borderRadius: '8px' }} />
              <video src="videos/auto_neutral.mp4" autoPlay loop muted playsInline controls style={{ width: '100%', borderRadius: '8px' }} />
            </Box>
          </Box>

          {/* Text Guided Exploration */}
          <Box sx={{ mb: 4 }}>
            <Typography variant='h5' component='h3' gutterBottom sx={{ color: 'primary.main', fontWeight: 500 }}>
              Text-Guided Generation
            </Typography>
            <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8, mb: 4 }}>
              Text-guided generation diverges from predictive, black-box paradigms by enforcing a direct mapping from the raw byte representation of input text directly to discrete codebook regions. Every text string becomes a deterministic and structurally unique "DNA fingerprint". Because adjacent bytes may map to biomechanically disconnected regions, our engine dynamically injects bridge regions using Dijkstra's algorithm over a precomputed plausibility graph. This routing securely connects abstract choreography blocks, forcing dynamic tempo changes and giving the text-driven dances a distinct rhythmic, kinetic signature unlike the homogeneous flow of the original dataset.
            </Typography>

            <Box sx={{ mb: 4 }}>
              <Typography variant='subtitle1' gutterBottom sx={{ fontWeight: 600, color: 'text.primary' }}>
                Guided by text "i love you"
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 2 }}>
                <video src="videos/text_ily_male.mp4" autoPlay loop muted playsInline controls style={{ width: '100%', borderRadius: '8px' }} />
                <video src="videos/text_ily_female.mp4" autoPlay loop muted playsInline controls style={{ width: '100%', borderRadius: '8px' }} />
                <video src="videos/text_ily_neutral.mp4" autoPlay loop muted playsInline controls style={{ width: '100%', borderRadius: '8px' }} />
              </Box>
            </Box>

            <Box sx={{ mb: 4 }}>
              <Typography variant='subtitle1' gutterBottom sx={{ fontWeight: 600, color: 'text.primary' }}>
                Guided by text "Life begins where fear ends"
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 2 }}>
                <video src="videos/text_lifebegins_male.mp4" autoPlay loop muted playsInline controls style={{ width: '100%', borderRadius: '8px' }} />
                <video src="videos/text_lifebegins_female.mp4" autoPlay loop muted playsInline controls style={{ width: '100%', borderRadius: '8px' }} />
                <video src="videos/text_lifebegins_neutral.mp4" autoPlay loop muted playsInline controls style={{ width: '100%', borderRadius: '8px' }} />
              </Box>
            </Box>

            <Box sx={{ mb: 2 }}>
              <Typography variant='subtitle1' gutterBottom sx={{ fontWeight: 600, color: 'text.primary' }}>
                Guided by text "Your Midas touch on the Chevy door"
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 2 }}>
                <video src="videos/text_midas_male.mp4" autoPlay loop muted playsInline controls style={{ width: '100%', borderRadius: '8px' }} />
                <video src="videos/text_midas_female.mp4" autoPlay loop muted playsInline controls style={{ width: '100%', borderRadius: '8px' }} />
                <video src="videos/text_midas_neutral.mp4" autoPlay loop muted playsInline controls style={{ width: '100%', borderRadius: '8px' }} />
              </Box>
            </Box>
          </Box>
        </Box>
        
        <Divider sx={{ borderColor: 'rgba(0,0,0,0.08)', mb: 6 }} />

        {/* Methodology Section */}
        <Box sx={{ mb: 6 }}>
          <Typography variant='h4' component='h2' gutterBottom sx={{ fontWeight: 600 }}>
            Methodology
          </Typography>

          <Box sx={{ mb: 4, mt: 3 }}>
            <Typography variant='h6' component='h3' gutterBottom sx={{ color: 'primary.main' }}>
              1. Data Aggregation and Standardization
            </Typography>
            <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8 }}>
              Our system aggregates human motion capture sequences represented via SMPL parametric models into a contiguous, multidimensional index. The raw parameters are flattened, and global coordinates are normalized using a subject-specific scale factor to ensure a robust, translation-invariant motion index mapped across the dataset.
            </Typography>
          </Box>
          
          <Box sx={{ mb: 4 }}>
            <Typography variant='h6' component='h3' gutterBottom sx={{ color: 'primary.main' }}>
              2. Feature Extraction and Quantization
            </Typography>
            <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8 }}>
              To establish an abstract representation of motion, we extract local relative kinematics and quantize the continuous human behavior space into a finite dictionary of stylistic regions. We apply local pose standardization and extract dynamic kinematics including local root velocities, local joint velocities, and binary foot contacts, yielding 278 features per frame. These features are then sliding-windowed into 20-frame chunks. We utilize <strong>Principal Component Analysis (PCA)</strong> to reduce the 5560 dimensions to a 64-dimensional space. Finally, <strong>K-Means clustering</strong> maps these projections to one of 256 localized centroid clusters, where every text frame is thus explicitly indexed to a motion codebook region in <MathJax inline>{"\\([0, 255]\\)"}</MathJax>.
            </Typography>
            <Box sx={{ mt: 3, mb: 2, textAlign: 'center' }}>
              <img src="codebook_grid.png" alt="Sample codebook regions" style={{ maxWidth: '100%', borderRadius: '8px' }} />
              <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary', fontStyle: 'italic' }}>
                Randomly sampled frames from each of the first 5 codebook regions. Note that the patterns shown are not perfect, as the quantization algorithm considers more than just the pose in the starting frame, and it is also challenging to describe the entire space of plausible human motions with just 256 regions. However, we can qualitatively observe similarities between frame poses from the same codebook region.
              </Typography>
            </Box>
          </Box>

          <Box sx={{ mb: 4 }}>
            <Typography variant='h6' component='h3' gutterBottom sx={{ color: 'primary.main' }}>
              3. Plausibility Graph Construction
            </Typography>
            <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8 }}>
              To govern biomechanically safe transitions between disjoint behaviors, we precompute a directed plausibility graph spanning the 256 codebook regions. An edge is defined based on a Motion Matching (MM) cost heuristic evaluating physical deviation (with a significant penalty applied to root velocity to prevent momentum shattering). We establish plausibility between regions when a valid path cost falls below a maximum threshold, saving edges iteratively evaluated from hundreds of candidate source to target frame pairings based on random sampling heuristics.
            </Typography>
            <Box sx={{ my: 2, overflowX: 'auto' }}>
              <MathJax>{"\\[C(x, x') = \\|p - p'\\|_2^2 + \\|\\dot{p} - \\dot{p}'\\|_2^2 + 10\\|\\dot{t} - \\dot{t}'\\|_2^2 + 2\\|\\Delta_{15} - \\Delta'_{15}\\|_2^2 + 2\\|\\Delta_{30} - \\Delta'_{30}\\|_2^2\\]"}</MathJax>
            </Box>
          </Box>

          <Box sx={{ mb: 4 }}>
            <Typography variant='h6' component='h3' gutterBottom sx={{ color: 'primary.main' }}>
              4. Text-Guided Generation and Interpolation
            </Typography>
            <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8 }}>
              At runtime, the engine uses the raw byte representation of input text. Since adjacent bytes may represent disjoint codebook regions, the system uses <strong>Dijkstra's algorithm</strong> on the precomputed plausibility graph to seamlessly inject bridge regions. When disjoint segments are appended, a <strong>Perlin Smootherstep polynomial</strong> guarantees <MathJax inline>{"\\(C^2\\)"}</MathJax> continuous easing over a blending window, and complex physical joints rely on <strong>Spherical Linear Interpolation (SLERP)</strong> to prevent structural limb collapse.
            </Typography>
          </Box>

          <Box sx={{ mb: 4 }}>
            <Typography variant='h6' component='h3' gutterBottom sx={{ color: 'primary.main' }}>
              5. Physics Smoothing and Correction
            </Typography>
            <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8 }}>
              Finally, synthesized arrays undergo post-processing to eliminate global vertical drift and prevent horizontal skating. A continuous dot-product alignment prevents boundary tearing by enforcing uniform hemispherical orientation across quaternions. A <strong>Savitzky-Golay quadratic filter</strong> smooths minor noise residuals, while an additive inverse matrix applies identical offset corrections directly to the root to neutralize lateral foot sliding when velocity falls below a minimum confidence threshold.
            </Typography>
          </Box>
        </Box>

      </Container>
    </Box>
    </MathJaxContext>
  );
};
export default App;
