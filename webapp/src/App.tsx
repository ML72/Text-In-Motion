import React from 'react';
import { Box, Typography, Button, Container, Divider } from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const App: React.FC = () => {
  return (
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

        {/* Overview Section */}
        <Box sx={{ mb: 6 }}>
          <Typography variant='h4' component='h2' gutterBottom sx={{ fontWeight: 600 }}>
            Overview
          </Typography>
          <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8 }}>
            Our research proposes a novel approach to choreographic generation by transforming the stylistic nuances of written prose directly into synthesized dance sequences. Rather than literal mapping, such as matching the word "walk" to a forward step, we encode the cadence, sentiment, and structural rhythm of textual inputs into corresponding motion sequences. Our pipeline bridges natural language processing and motion synthesis to explore the connection between spoken language and physical expression.
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
              Autonomous exploration is where the algorithm operates unconstrained, synthesizing new dances from individual moves entirely from scratch. Drawing from a diverse library of dance styles, the choreography continuously shifts and evolves. This dynamic blending is driven by a built-in novelty penalty that naturally incentivizes the algorithm to seek out unseen frames and new movements as the performance progresses. Free from the rigid region constraints that might otherwise limit algorithmic expressivity, the resulting dance unfolds as an organically smooth, abstract art form.
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
              Text Guided Visualization
            </Typography>
            <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8, mb: 4 }}>
              Text guided visualization interprets the exact UTF-8 encoding of an input text as a rigid string of quantization regions to track, effectively giving each sentence its own unique "motion DNA". While these strict textual constraints anchor the performance, they can sometimes cause abrupt glitches or unnatural transitions between distant stylistic states in the generated movements. Rather than being seen purely as artifacts, these moments can be interpreted as the unique, artistic fingerprints of the text's inherent rhythm. This approach differs from typical methods by deliberately setting aside the literal semantic meaning of words. By instead building a new discrete vocabulary and mapping it directly onto movement, our method creates an intriguing, abstract exploration of prose turning into motion.
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
              1. Feature Extraction & Dimensionality Reduction
            </Typography>
            <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8 }}>
              We extract high-dimensional raw motion features over a temporal window <InlineMath math={String.raw`W`} /> (defaulting to 20 frames) representing the "stylistic future" of each frame. Using the <strong>SMPL body model</strong>, we extract local behavior features including joint poses, root velocities, and foot contacts, intentionally discarding world-space position variations for translation-invariance. To produce a more robust latent representation, we apply <strong>Principal Component Analysis (PCA)</strong> to project the frame-wise motion features into a 64-dimensional space:
            </Typography>
            <Box sx={{ my: 2, overflowX: 'auto' }}>
              <BlockMath math={String.raw`f_{PCA} = W_{PCA}^T (f_{raw} - \mu)`} />
            </Box>
          </Box>

          <Box sx={{ mb: 4 }}>
            <Typography variant='h6' component='h3' gutterBottom sx={{ color: 'primary.main' }}>
              2. Motion Quantization
            </Typography>
            <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8 }}>
              Continuous 64D motion embeddings are discretized into a finite dictionary of 256 discrete regions to build a robust stylistic codebook. Using <strong>K-Means clustering</strong>, for each frame <InlineMath math={String.raw`t`} />, the projected vector is mapped to the nearest centroid <InlineMath math={String.raw`c^{(t)}`} /> within the codebook vocabulary <InlineMath math={String.raw`\mathcal{C}`} />.
            </Typography>
          </Box>

          <Box sx={{ mb: 4 }}>
            <Typography variant='h6' component='h3' gutterBottom sx={{ color: 'primary.main' }}>
              3. Plausibility Graph
            </Typography>
            <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8 }}>
              To ensure smooth transitions between discrete action states, we compute an offline dynamic transition plausibility graph. The transition likelihood is derived from edge weights based on the state similarity mapping cost between poses and velocities to enforce motion continuity:
            </Typography>
            <Box sx={{ my: 2, overflowX: 'auto' }}>
              <BlockMath math={String.raw`\text{Cost}(i, j) = \alpha \|p_i - p_j\|_2^2 + \beta \|v_i - v_j\|_2^2`} />
            </Box>
          </Box>

          <Box sx={{ mb: 4 }}>
            <Typography variant='h6' component='h3' gutterBottom sx={{ color: 'primary.main' }}>
              4. Text-Guided Generation
            </Typography>
            <Typography variant='body1' sx={{ color: 'text.secondary', lineHeight: 1.8 }}>
              Using <strong>Motion Matching</strong>, the generative tracking engine navigates the stylistic codebook via a user-provided textual target sequence. Text inputs are trimmed and directly converted into a string of bytes (0-255) corresponding to codebook regions. If a direct transition to a targeted region is physically implausible within the search window, the system uses <strong>Dijkstra&apos;s algorithm</strong> on the plausibility graph to seamlessly inject bridge regions, ensuring continuous tracking aligned with the rhythmic constraints of the text and human kinematics.
            </Typography>
          </Box>
        </Box>

      </Container>
    </Box>
  );
};
export default App;
