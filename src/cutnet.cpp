/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "cutnet.h"

#include <algorithm>

bool CutNet::shouldCut(int depth, int dStaticEval, bool ttHit, int dttDepth, int correctionValue, bool improving, bool dSimpleEval) {
	// Input features
	float input[7] = {
		std::clamp(static_cast<float>(depth) / 3.0f, 0.0f, 1.0f),
		std::clamp(static_cast<float>(dStaticEval) / 300.0f, -1.0f, 1.0f),
		std::clamp(static_cast<float>(ttHit ? 1.0f : 0.0f), 0.0f, 1.0f),
		std::clamp(static_cast<float>(dttDepth) / 5.0f, -1.0f, 1.0f),
		std::clamp(static_cast<float>(correctionValue) / 20.0f, -1.0f, 1.0f),
		std::clamp(static_cast<float>(improving ? 1.0f : 0.0f), 0.0f, 1.0f),
		std::clamp(static_cast<float>(dSimpleEval) / 300.0f, -1.0f, 1.0f)
	};

	// L1
	float fc1[32];
	for (int i = 0; i < 32; i++) {
		fc1[i] = CutNet::fc1_b[i];
		for (int j = 0; j < 7; j++) {
			fc1[i] += CutNet::fc1_w[i][j] * input[j];
		}
		if (fc1[i] < 0.0f) fc1[i] = 0.0f;
	}

	// L2
	float fc2 = CutNet::fc2_b;
	for (int i = 0; i < 32; i++) {
		fc2 += CutNet::fc2_w[i] * fc1[i];
	}

	return fc2 > 2.2f; // Corresponds to sigmoid 0.9
}