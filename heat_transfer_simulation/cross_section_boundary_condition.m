function leftTemp = cross_section_boundary_condition(~, state)

MAXVAL = 1;
MINVAL = 0;

offset_time = 0.5;
lin_slope = (MAXVAL-MINVAL) / offset_time;
exp_slope = 0.75;
exp_slope2 = 10;


if(isnan(state.time))
  leftTemp = NaN;
elseif(state.time <= offset_time)
  leftTemp = -(MAXVAL-MINVAL) * exp(-(state.time) * exp_slope2) + MAXVAL;
%   leftTemp = state.time * lin_slope + MINVAL;
else
  leftTemp = (MAXVAL-MINVAL) * exp(-(state.time-offset_time) * exp_slope) + MINVAL;
end
end