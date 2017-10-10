%     Copyright (C) 2017  Matthieu Pizenberg, Axel Carlier
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.


function [ segmentation, nb_iter ] = segment( img, bg_constraint, fg_constraint, opts )
% Segment an image into foreground (fg) and background (bg).
% Constraints are provided for foreground and background.
%
% Syntax #####################
%
% [ segmentation, nb_iter ] = segment( img, bg_constraint, fg_constraint, opts );
%
% Description ################
%
% img: m x n x 3 array. RGB color image.
% bg_constraint: { type: String, mask: m * n logical array }
%     type = 'none' | 'soft' | 'hard'
%     A soft constraint applies only at initialization, whereas a hard constraint
%     is reapplied at each iteration, and at the end of the algorithm.
% fg_constraint: same as bg_constraint.
% opts: { nb_classes: int, max_nb_iter: int, diff_thresh: float, jump_cost: float, beta: float }


% Initialization.
if nargin < 4
	opts = defaultParameters();
end
nb_classes = opts.nb_classes;
if nargin < 3
	fg_constraint.type = 'none';
end
img = double( img );
is_bg_none = strcmp( 'none', bg_constraint.type );
is_fg_none = strcmp( 'none', fg_constraint.type );
is_bg_hard = strcmp( 'hard', bg_constraint.type );
is_fg_hard = strcmp( 'hard', fg_constraint.type );
assert( ~ ( is_bg_none && is_fg_none ), 'BG or FG must be different from none' );

% If one constraint is 'none', initialize it with the opposite mask.
if is_fg_none
	fg_constraint.mask = ~ bg_constraint.mask;
elseif is_bg_none
	bg_constraint.mask = ~ fg_constraint.mask;
end

% Compute static graph cut costs.
[ smoothness_cost, v_c, h_c ] = graphCutStaticCosts( img, opts.jump_cost, opts.beta );

% Initialize loop values.
prev_seg = ~ bg_constraint.mask;
bg_values = reshape( img( bg_constraint.mask(:,:,[1,1,1]) ), [], 3 );
fg_values = reshape( img( fg_constraint.mask(:,:,[1,1,1]) ), [], 3 );
means_bg = [];
means_fg = [];

% Main iterations loop.
for iter = 1 : opts.max_nb_iter

	% Verify that both initial masks are of sufficient size (> nb_classes).
	if size(bg_values, 1) < nb_classes || size(fg_values, 1) < nb_classes
		nb_iter = iter - 1;
		segmentation = prev_seg;
		return
	end

	% (Steps 1 & 2) Compute negative log likelihood of FG and BG with GMM.
	[ neg_logl_bg, means_bg ] = GMM( img, bg_values, nb_classes, means_bg );
	[ neg_logl_fg, means_fg ] = GMM( img, fg_values, nb_classes, means_fg );

	% Apply hard prior constraints on likelihoods.
	if is_bg_hard
		neg_logl_fg( bg_constraint.mask ) = max( neg_logl_fg(:) );
	end
	if is_fg_hard
		neg_logl_bg( fg_constraint.mask ) = max( neg_logl_bg(:) );
	end
	
	% (Step 3) Compute graph cut.
	data_cost = cat( 3, neg_logl_bg, neg_logl_fg );
	segmentation = logical ...
		( computeGraphCut( data_cost, smoothness_cost, v_c, h_c, prev_seg ) );

	% Apply hard prior constraints on result segmentation mask.
	if is_bg_hard
		segmentation( bg_constraint.mask ) = false;
	end
	if is_fg_hard
		segmentation( fg_constraint.mask ) = true;
	end

	% Avoid preparing for next loop if it is the last one.
	if iter == opts.max_nb_iter
		break;
	end

	% End if current segmentation is similar to previous one.
	if sum( segmentation ~= prev_seg ) < opts.diff_thresh * numel( segmentation )
		break;
	end
	
	% Update loop values.
	prev_seg = segmentation;
	bg_values = reshape( img( ~ segmentation(:,:,[1,1,1]) ), [], 3 );
	fg_values = reshape( img( segmentation(:,:,[1,1,1]) ), [], 3 );

end

% Get the number of completed iterations.
nb_iter = iter;

end


%###################################### Helper functions


function opts = defaultParameters()
% Default parameters for GrabCut.
	opts.nb_classes = 5;
	opts.max_nb_iter = 10;
	opts.diff_thresh = 5e-4;
	% Graph Cut options
	opts.jump_cost = 50;
	opts.beta = 0.5;
end


function [ neg_logl, means ] = GMM( img, train_values, nb_classes, means_init )
% Compute negative log likelihood of pixels in img,
% by building a Gaussian Mixture Model (GMM) based on the given values.
% nb_values must be >= nb_classes.
% means_init is optional. If present, it must be empty or of size nb_classes x 3

	% Assert preconditions.
	nb_values = size( train_values, 1 );
	assert( nb_classes >= 1, 'Needs at least 1 class' );
	assert( nb_values >= nb_classes, 'Not enough values' );

	% Initialize GMM with k-means.
	w = warning( 'off', 'all' );
	if nargin == 4 && ~ isempty( means_init )
		assert( size( means_init, 1 ) == nb_classes, 'Wrong number of initial means' );
		[ cluster_ids, means ] = ...
			kmeans( train_values, nb_classes, 'start', means_init, 'MaxIter', 40 );
	else
		[ cluster_ids, means ] = kmeans( train_values, nb_classes, 'MaxIter', 40 );
	end
	warning( w );

	% Compute GMM weights priors based on occurences of each class.
	weights_prior = histcounts( cluster_ids, 1:nb_classes+1 ) ./ nb_values;

	% Compute the covariance matrix for each class.
	% regularized cov to avoid ill conditionning,
	% that may happen if a class contains identical pixels.
	reg_cov = 1e-10 * eye(3);
	sigmas = zeros( 3, 3, nb_classes );
	det_sigmas = zeros(1, nb_classes );
	for k = 1:nb_classes
		current_cluster_ind = ( cluster_ids == k );
		% Special case if current cluster contains only one observation otherwise,
		% cov will not build a covariance matrix but variance of R, G, B.
		if length( current_cluster_ind ) == 1
			sigma = reg_cov;
		else
			sigma = reg_cov + cov( train_values( current_cluster_ind, : ) );
		end
		sigmas( :, :, k ) = sigma;
		det_sigmas( k ) = det( sigma );
	end

	% Build GMM.
	% https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	% https://fr.mathworks.com/help/stats/gmdistribution-class.html
	% gmm = gmdistribution( means, sigmas );
	% pdf = gmm.pdf( reshape( img, [], 3 ) );
	% neg_logl_each = bsxfun( @minus, log(weights_prior), 0.5 * log(pdf) );
	% neg_logl = reshape( min( neg_logl_each, [], 2 ), size(img,1), size(img,2) );

	% The code below does the same thing as commented just above,
	% just slightly faster, so let's save few micro seconds!
	neg_logl_each = zeros( size(img,1) * size(img,2), nb_classes );
	img_values = reshape( img, [], 3 );
	for k = 1:nb_classes
		k_dist = bsxfun( @minus, img_values, means(k,:) );
		neg_logl_each(:,k) = ...
			0.5 * ( sum( ( k_dist / sigmas(:,:,k) ) .* k_dist, 2 ) ...
			+ log( det_sigmas(k) ) ) ...
			- log( weights_prior(k) );
	end
	neg_logl = reshape( min( neg_logl_each, [], 2 ), size(img,1), size(img,2) );

end


function [ smoothness_cost, v_c, h_c ] = graphCutStaticCosts( img, jump_cost, beta )
% Initialize GraphCut costs parameters only based on the image.

	% Use jump_cost from one label to the other for smoothness_cost.
	smoothness_cost = [ 0 jump_cost; jump_cost 0 ];

	% Get the image gradient (squared).
	grad_v = img( 2:end, :, : ) - img( 1:end-1, :, : );
	grad_h = img( :, 2:end, : ) - img( :, 1:end-1, : );
	grad_v_sqr = sum( grad_v.^2, 3 );
	grad_h_sqr = sum( grad_h.^2, 3 );

	% Use gradient to compute the graph's inter-pixels weights.
	v_c = exp( -beta / mean( grad_v_sqr(:) ) * grad_v_sqr );
	h_c = exp( -beta / mean( grad_h_sqr(:) ) * grad_h_sqr );

	v_c = [ v_c; zeros( 1, size(v_c,2) ) ];
	h_c = [ h_c, zeros( size(h_c,1), 1 ) ];

end


function labels = computeGraphCut( data_cost, smoothness_cost, v_c, h_c, labels )
% Use the GraphCut functions provided by GCMex to compute new labels.

	handle = GraphCut( 'open', data_cost, smoothness_cost, v_c, h_c );
	handle = GraphCut( 'set', handle, labels );
	[ handle, labels ] = GraphCut( 'expand', handle );
	GraphCut( 'close', handle );

end
