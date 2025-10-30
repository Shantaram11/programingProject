package main

import (
	"math"
	"math/rand"
)

// function to initialize an ecosystem with initial populations and families.
func InitializeEcosystem() Ecosystem {
	w := Ecosystem_Width
	rabbit := Species{Name: "rabbit", Type: "prey", GrowthRate: GrowthRate_Rabbit}
	sheep := Species{Name: "sheep", Type: "prey", GrowthRate: GrowthRate_Sheep}
	deer := Species{Name: "deer", Type: "prey", GrowthRate: GrowthRate_Deer}
	wolf := Species{Name: "wolf", Type: "predator", GrowthRate: GrowthRate_Wolf}

	var families []Family
	add := func(total int, sp Species) {
		sizes := randomPartition(total, initial_family_number, Smallest_Family_Size)
		for _, s := range sizes {
			pos := OrderedPair{rand.Float64() * w, rand.Float64() * w}
			kx := rand.Float64()*2 - 1
			ky := rand.Float64()*2 - 1
			families = append(families, Family{
				Size:              s,
				MovementSpeed:     OrderedPair{rand.Float64()*2 - 1, rand.Float64()*2 - 1},
				Position:          pos,
				MovementDirection: OrderedPair{kx, ky},
				species:           sp,
			})
		}
	}
	add(initial_Population_Rabbit, rabbit)
	add(initial_Population_Sheep, sheep)
	add(initial_Population_Deer, deer)
	add(initial_Population_Wolf, wolf)

	var humans []Human
	for i := 0; i < initial_Population_Human; i++ {
		pos := OrderedPair{rand.Float64() * w, rand.Float64() * w}
		kx := rand.Float64()*2 - 1
		ky := rand.Float64()*2 - 1
		humans = append(humans, Human{
			MovementSpeed:     OrderedPair{rand.Float64()*2 - 1, rand.Float64()*2 - 1},
			Position:          pos,
			MovementDirection: OrderedPair{kx, ky},
		})
	}

	return Ecosystem{
		Families:          families,
		Humans:            humans,
		Population_rabbit: initial_Population_Rabbit,
		Population_sheep:  initial_Population_Sheep,
		Population_deer:   initial_Population_Deer,
		Population_wolf:   initial_Population_Wolf,
		Population_human:  initial_Population_Human,
		width:             w,
	}
}

// Help function to initialize family sizes randomly
func randomPartition(total, k, min int) []int {
	if k <= 0 || total < k*min {
		return []int{}
	}
	parts := make([]int, k)
	remain := total - k*min
	cuts := make([]int, k-1)
	for i := range cuts {
		cuts[i] = rand.Intn(remain + 1)
	}
	cuts = append(cuts, 0)
	cuts = append(cuts, remain)
	for i := 0; i+1 < len(cuts); i++ {
		a, b := cuts[i], cuts[i+1]
		if a > b {
			a, b = b, a
		}
		parts[i] = (b - a) + min
	}
	sum := 0
	for i := 0; i < k-1; i++ {
		sum += parts[i]
	}
	parts[k-1] = total - sum
	return parts
}

// function to update populations based on growth rates
func UpdatePopulations(ecosystem Ecosystem) Ecosystem {
	for i := range ecosystem.Families {
		gr := ecosystem.Families[i].species.GrowthRate // passive growth rate
		// prey gets additional growth from plants, should be updated in next step
		if ecosystem.Families[i].species.Type == "prey" {
			gr += PlantCoefficient
		}

		//UPDATE!!! also need to update growth rate based on interactions with other species

		//

		size := ecosystem.Families[i].Size
		newSize := size + int(math.Round(float64(size)*gr))
		if newSize < 0 {
			newSize = 0
		}
		ecosystem.Families[i].Size = newSize

		compacted := ecosystem.Families[:0]
		for _, f := range ecosystem.Families {
			if f.Size > 0 {
				compacted = append(compacted, f)
			}
		}
		ecosystem.Families = compacted
	}

	r, s, d, w := 0, 0, 0, 0
	for _, f := range ecosystem.Families {
		switch f.species.Name {
		case "rabbit":
			r += f.Size
		case "sheep":
			s += f.Size
		case "deer":
			d += f.Size
		case "wolf":
			w += f.Size
		}
	}
	ecosystem.Population_rabbit = r
	ecosystem.Population_sheep = s
	ecosystem.Population_deer = d
	ecosystem.Population_wolf = w
	return ecosystem
}

// function to merge small family with someone nearby
func MergeFamilies(ecosystem Ecosystem) Ecosystem {
	f := ecosystem.Families
	for i := 0; i < len(f); {
		if f[i].Size < Smallest_Family_Size {
			merged := false
			for j := 0; j < len(f); j++ {
				if i == j || f[i].species.Name != f[j].species.Name {
					continue
				}
				if distance(f[i].Position, f[j].Position) <= Merging_Threshould {
					f[j].Size += f[i].Size
					f[i] = f[len(f)-1]
					f = f[:len(f)-1]
					merged = true
					break
				}
			}
			if merged {
				continue
			}
		}
		i++
	}
	ecosystem.Families = f
	return ecosystem
}

func distance(a, b OrderedPair) float64 {
	return math.Hypot(a.x-b.x, a.y-b.y)
}
