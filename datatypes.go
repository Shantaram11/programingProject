package main

type Ecosystem struct {
	Families          []Family
	Humans            []Human
	Population_rabbit int
	Population_sheep  int
	Population_deer   int
	Population_wolf   int
	Population_human  int
	width             float64
}
type Species struct {
	Name              string
	Type              string
	GrowthRate        float64
	ContactGrowthRate float64
}

type Family struct {
	Size              int
	MovementSpeed     OrderedPair
	Position          OrderedPair
	MovementDirection OrderedPair
	species           Species
}

type Human struct {
	MovementSpeed     OrderedPair
	Position          OrderedPair
	MovementDirection OrderedPair
}

type OrderedPair struct {
	x float64
	y float64
}

var initial_family_number = 3
var initial_Population_Rabbit = 100
var initial_Population_Sheep = 50
var initial_Population_Deer = 30
var initial_Population_Wolf = 20
var initial_Population_Human = 2

const GrowthRate_Rabbit = 0.5
const GrowthRate_Sheep = 0.3
const GrowthRate_Deer = 0.2
const GrowthRate_Wolf = -0.1
const PlantCoefficient = 0.05
const Merging_Threshould = 20.0
const Smallest_Family_Size = 5
const Ecosystem_Width = 500.0
