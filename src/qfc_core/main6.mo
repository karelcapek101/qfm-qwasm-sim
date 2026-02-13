// Berry-Keating Sim Extension
public query func berry_keating_sim() : async {
  variant {
    evals : [Float];
    mean_spacing : Float;
    pred_log_t : Float;
    match_ratio : Float;
  };
  let N : Nat = 128;
  let evals : [Float] = Array.tabulate<Float>(N, func (k) {
    let k_float = Float.fromIntWrap(k + 1);
    (k_float + 0.5) * 3.14159 + 1.0 / k_float  // Approx Im(ρ_k)
  });
  var sum_sp : Float = 0.0;
  var count : Nat = 0;
  for (i in Iter.range(0, Nat.sub(evals.size(), 1))) {
    let sp = Float.sub(Array.get(evals, i+1), Array.get(evals, i));
    sum_sp += sp;
    count += 1;
  };
  let mean_sp : Float = sum_sp / Float.fromIntWrap(count);
  let T : Float = Array.get(evals, evals.size() - 1);
  let pred : Float = Float.log(T) / (2.0 * 3.14159);
  let ratio : Float = mean_sp / pred;
  #ok({ evals = evals; mean_spacing = mean_sp; pred_log_t = pred; match_ratio = ratio })
};
